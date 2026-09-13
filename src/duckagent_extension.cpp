#define DUCKDB_EXTENSION_MAIN

#include "duckagent_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// agent.cpp - local LLM agent runtime (see CMakeLists.txt FetchContent block)
#include "agent.h"
#include "error.h"
#include "model.h"
#include "llama.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace duckdb {

//===--------------------------------------------------------------------===//
// ai_enrich: bind data
//===--------------------------------------------------------------------===//
// Holds the field names parsed out of the `schema` argument at bind time.
// v1 only supports the "simple schema" form: a flat JSON array of field
// names, e.g. ["industry","headquarters_country","year_founded"], and every
// field is generated as VARCHAR. Typed/nested schema is a later addition.
struct AIEnrichBindData : public FunctionData {
	explicit AIEnrichBindData(vector<string> field_names_p) : field_names(std::move(field_names_p)) {
	}

	vector<string> field_names;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AIEnrichBindData>(field_names);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<AIEnrichBindData>();
		return field_names == other.field_names;
	}
};

//===--------------------------------------------------------------------===//
// Minimal parser for the "simple schema" form: ["a","b","c"]
//===--------------------------------------------------------------------===//
// This intentionally does not depend on the json extension - it is a
// hand-rolled parser for exactly one shape: a JSON array of string
// literals. Anything else (the advanced typed/nested schema) throws for
// now, so behavior is explicit rather than silently wrong.
static vector<string> ParseSimpleSchemaFields(const string &schema_json) {
	vector<string> fields;

	idx_t i = 0;
	auto n = schema_json.size();

	auto SkipWhitespace = [&]() {
		while (i < n && isspace(static_cast<unsigned char>(schema_json[i]))) {
			i++;
		}
	};

	SkipWhitespace();
	if (i >= n || schema_json[i] != '[') {
		throw InvalidInputException(
		    "ai_enrich: schema must be a JSON array of field names, e.g. [\"industry\",\"year_founded\"]. "
		    "Typed/nested schemas are not yet supported.");
	}
	i++; // consume '['

	SkipWhitespace();
	if (i < n && schema_json[i] == ']') {
		throw InvalidInputException("ai_enrich: schema array must contain at least one field name");
	}

	while (i < n) {
		SkipWhitespace();
		if (i >= n || schema_json[i] != '"') {
			throw InvalidInputException("ai_enrich: expected a quoted field name in schema at position %llu",
			                             (unsigned long long)i);
		}
		i++; // consume opening quote
		string field;
		while (i < n && schema_json[i] != '"') {
			// no escape-sequence handling needed for v1 field names
			field += schema_json[i];
			i++;
		}
		if (i >= n) {
			throw InvalidInputException("ai_enrich: unterminated string in schema");
		}
		i++; // consume closing quote
		if (field.empty()) {
			throw InvalidInputException("ai_enrich: schema field names cannot be empty");
		}
		fields.push_back(field);

		SkipWhitespace();
		if (i < n && schema_json[i] == ',') {
			i++;
			continue;
		}
		if (i < n && schema_json[i] == ']') {
			i++;
			break;
		}
		throw InvalidInputException("ai_enrich: malformed schema, expected ',' or ']' at position %llu",
		                             (unsigned long long)i);
	}

	return fields;
}

//===--------------------------------------------------------------------===//
// ai_enrich: bind
//===--------------------------------------------------------------------===//
// Resolves the return type (STRUCT with one VARCHAR field per schema entry)
// at bind time, since `schema` must be a constant argument - matching the
// Databricks contract that schema is a STRING literal.
static unique_ptr<FunctionData> AIEnrichBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < 2) {
		throw BinderException("ai_enrich requires at least (content, schema) arguments");
	}

	if (!arguments[1]->IsFoldable()) {
		throw BinderException("ai_enrich: the 'schema' argument must be a constant string literal");
	}

	Value schema_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (schema_value.IsNull()) {
		throw BinderException("ai_enrich: 'schema' cannot be NULL");
	}
	auto schema_str = schema_value.ToString();

	auto field_names = ParseSimpleSchemaFields(schema_str);

	child_list_t<LogicalType> struct_children;
	for (auto &field : field_names) {
		struct_children.emplace_back(field, LogicalType::VARCHAR);
	}
	bound_function.SetReturnType(LogicalType::STRUCT(struct_children));

	return make_uniq<AIEnrichBindData>(std::move(field_names));
}

//===--------------------------------------------------------------------===//
// agent.cpp model loading
//===--------------------------------------------------------------------===//
// llama_log_set() forwards internally to ggml_log_set() as well, so a
// single no-op callback silences everything from model-load progress and
// KV cache messages down to backend-level logs like Metal's
// "ggml_metal_free: deallocating" - all of which otherwise print straight
// to stderr with no per-call way to suppress them.
// TODO: make this configurable (e.g. DUCKAGENT_VERBOSE=1) if the logs turn
// out to be useful for debugging later.
static void SilentGgmlLogCallback(ggml_log_level /*level*/, const char * /*text*/, void * /*user_data*/) {
}

// Weights are the expensive part (the GGUF file itself) and are shared
// across every ai_enrich call in the process. Each call site gets its own
// Model (own KV cache/context) via Model::create_with_weights, since
// contexts aren't safe to share across concurrent DuckDB threads.
//
// TODO: replace the DUCKAGENT_MODEL_PATH env var with a proper DuckDB
// extension setting (e.g. `SET duckagent_model_path = '...'`) once that
// wiring exists - env var is a placeholder to unblock local development.
static std::shared_ptr<agent_cpp::ModelWeights> GetSharedModelWeights() {
	static std::mutex weights_mutex;
	static std::shared_ptr<agent_cpp::ModelWeights> weights;

	std::lock_guard<std::mutex> lock(weights_mutex);
	if (weights) {
		return weights;
	}

	llama_log_set(SilentGgmlLogCallback, nullptr);

	const char *model_path_env = std::getenv("DUCKAGENT_MODEL_PATH");
	if (!model_path_env || string(model_path_env).empty()) {
		throw InvalidInputException(
		    "ai_enrich: no local model configured. Set the DUCKAGENT_MODEL_PATH environment variable to "
		    "a local GGUF model file before calling ai_enrich. (TODO: replace with a proper DuckDB "
		    "extension setting.)");
	}

	try {
		weights = agent_cpp::ModelWeights::create(model_path_env);
	} catch (const agent_cpp::ModelError &e) {
		throw IOException("ai_enrich: failed to load model from '%s': %s", model_path_env, e.what());
	}
	return weights;
}

//===--------------------------------------------------------------------===//
// GBNF grammar generation
//===--------------------------------------------------------------------===//
// Builds a grammar that constrains generation to exactly
// {"field1":"...", "field2":"...", ...} in schema order, all string values.
// The `string`/`ws` rules are lifted directly from llama.cpp's own
// grammars/json.gbnf reference grammar.
static string EscapeGbnfLiteral(const string &field_name) {
	string escaped;
	escaped.reserve(field_name.size());
	for (char c : field_name) {
		if (c == '"' || c == '\\') {
			escaped += '\\';
		}
		escaped += c;
	}
	return escaped;
}

static const char *kEnrichGrammarStringAndWsRules =
    "string ::=\n"
    "  \"\\\"\" (\n"
    "    [^\"\\\\\\x7F\\x00-\\x1F] |\n"
    "    \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n"
    "  )* \"\\\"\" ws\n"
    "\n"
    "ws ::= | \" \" | \"\\n\" [ \\t]{0,20}\n";

static string BuildEnrichGrammar(const vector<string> &field_names) {
	string root = "root ::= \"{\" ws";
	for (idx_t i = 0; i < field_names.size(); i++) {
		if (i > 0) {
			root += " \",\" ws";
		}
		root += " \"\\\"" + EscapeGbnfLiteral(field_names[i]) + "\\\":\" ws string";
	}
	root += " \"}\" ws\n\n";
	return root + kEnrichGrammarStringAndWsRules;
}

static string BuildEnrichInstructions(const vector<string> &field_names) {
	string instructions =
	    "You are a data enrichment assistant. Given a row of input data, infer the value of each of the "
	    "following fields using your own knowledge: ";
	for (idx_t i = 0; i < field_names.size(); i++) {
		if (i > 0) {
			instructions += ", ";
		}
		instructions += field_names[i];
	}
	instructions +=
	    ". Give your best estimate for every field rather than leaving it blank, but do not fabricate "
	    "implausible specifics - if you are unsure, provide a reasonable general answer.";
	return instructions;
}

//===--------------------------------------------------------------------===//
// Minimal parser for the flat JSON object our grammar guarantees:
// {"a":"b","c":"d"}. Not a general JSON parser - deliberately only handles
// the shape the grammar can produce.
//===--------------------------------------------------------------------===//
static bool TryParseFlatJSONObjectStrings(const string &json, std::unordered_map<string, string> &out) {
	idx_t i = 0;
	auto n = json.size();

	auto SkipWhitespace = [&]() {
		while (i < n && isspace(static_cast<unsigned char>(json[i]))) {
			i++;
		}
	};

	auto ParseString = [&](string &value) -> bool {
		SkipWhitespace();
		if (i >= n || json[i] != '"') {
			return false;
		}
		i++;
		value.clear();
		while (i < n && json[i] != '"') {
			char c = json[i];
			if (c == '\\') {
				i++;
				if (i >= n) {
					return false;
				}
				switch (json[i]) {
				case '"':
					value += '"';
					break;
				case '\\':
					value += '\\';
					break;
				case '/':
					value += '/';
					break;
				case 'n':
					value += '\n';
					break;
				case 't':
					value += '\t';
					break;
				case 'r':
					value += '\r';
					break;
				case 'b':
					value += '\b';
					break;
				case 'f':
					value += '\f';
					break;
				case 'u':
					// TODO: decode \uXXXX properly; skip for v1.
					i += 4;
					break;
				default:
					value += json[i];
				}
				i++;
			} else {
				value += c;
				i++;
			}
		}
		if (i >= n) {
			return false;
		}
		i++; // consume closing quote
		return true;
	};

	SkipWhitespace();
	if (i >= n || json[i] != '{') {
		return false;
	}
	i++;
	SkipWhitespace();
	if (i < n && json[i] == '}') {
		return true; // empty object
	}

	while (i < n) {
		string key;
		if (!ParseString(key)) {
			return false;
		}
		SkipWhitespace();
		if (i >= n || json[i] != ':') {
			return false;
		}
		i++;
		string value;
		if (!ParseString(value)) {
			return false;
		}
		out[key] = value;

		SkipWhitespace();
		if (i < n && json[i] == ',') {
			i++;
			SkipWhitespace();
			continue;
		}
		if (i < n && json[i] == '}') {
			return true;
		}
		return false;
	}
	return false;
}

//===--------------------------------------------------------------------===//
// Sanity check for generated field values
//===--------------------------------------------------------------------===//
// Grammar-constrained decoding only guarantees syntactic validity (a legal
// JSON string), not that the *content* is sane. A weak/small model at
// temp=0 (fully greedy) can walk into a degenerate path for a fact it
// doesn't confidently know and produce junk that is still grammar-legal -
// stray structural characters, echoed prompt fragments, etc. This is a
// cheap, model-agnostic heuristic to catch that class of failure: a
// genuine short factual answer ("San Francisco", "Aerospace") should never
// contain JSON/HTML structural characters or come back empty.
static bool LooksGarbled(const string &value) {
	if (value.empty()) {
		return true;
	}
	for (char c : value) {
		if (c == '<' || c == '>' || c == '{' || c == '}') {
			return true;
		}
	}
	return false;
}

//===--------------------------------------------------------------------===//
// Runs one full agent turn for a row and parses the result.
// Returns false if generation threw or the response didn't parse as the
// flat JSON object our grammar guarantees - callers still need to check
// individual field values with LooksGarbled() even when this returns true.
//===--------------------------------------------------------------------===//
static bool GenerateEnrichmentFields(agent_cpp::Agent &agent, const string &content,
                                      std::unordered_map<string, string> &out) {
	std::vector<common_chat_msg> messages;
	common_chat_msg user_msg;
	user_msg.role = "user";
	user_msg.content = content;
	messages.push_back(std::move(user_msg));

	string response_text;
	try {
		response_text = agent.run_loop(messages);
	} catch (const agent_cpp::Error &e) {
		return false;
	}
	return TryParseFlatJSONObjectStrings(response_text, out);
}


// Holds a fully-configured Agent (model + grammar baked in for this call
// site's schema, no tools in v1) so it's built once and reused across every
// DataChunk processed on this thread, not reloaded per row or per chunk.
// Retry generation uses a nonzero temperature specifically to escape a
// greedy (temp=0) decode that walked into a degenerate path - a different
// sampling path is often enough to dodge the same dead end. Only used for
// fields that fail LooksGarbled() on the primary (temp=0) attempt, so
// well-behaved rows never pay this extra generation cost.
static constexpr float kEnrichRetryTemperature = 0.7F;

struct AIEnrichLocalState : public FunctionLocalState {
	std::shared_ptr<agent_cpp::Agent> agent;
	std::shared_ptr<agent_cpp::Agent> retry_agent;
};

static unique_ptr<FunctionLocalState> AIEnrichInitLocalState(ExpressionState &/*state*/,
                                                              const BoundFunctionExpression &/*expr*/,
                                                              FunctionData *bind_data_p) {
	auto &info = bind_data_p->Cast<AIEnrichBindData>();

	auto weights = GetSharedModelWeights();
	auto grammar = BuildEnrichGrammar(info.field_names);
	string instructions = BuildEnrichInstructions(info.field_names);

	agent_cpp::ModelConfig primary_config;
	primary_config.grammar = grammar;
	primary_config.grammar_root = "root";
	primary_config.temp = 0.0F; // deterministic: field extraction, not creative writing

	agent_cpp::ModelConfig retry_config;
	retry_config.grammar = grammar;
	retry_config.grammar_root = "root";
	retry_config.temp = kEnrichRetryTemperature;

	std::shared_ptr<agent_cpp::Model> model;
	std::shared_ptr<agent_cpp::Model> retry_model;
	try {
		model = agent_cpp::Model::create_with_weights(weights, primary_config);
		retry_model = agent_cpp::Model::create_with_weights(weights, retry_config);
	} catch (const agent_cpp::ModelError &e) {
		throw IOException("ai_enrich: failed to initialize model context: %s", e.what());
	}

	auto local_state = make_uniq<AIEnrichLocalState>();
	local_state->agent = std::make_shared<agent_cpp::Agent>(
	    std::move(model), std::vector<std::unique_ptr<agent_cpp::Tool>>{},
	    std::vector<std::unique_ptr<agent_cpp::Callback>>{}, instructions);
	local_state->retry_agent = std::make_shared<agent_cpp::Agent>(
	    std::move(retry_model), std::vector<std::unique_ptr<agent_cpp::Tool>>{},
	    std::vector<std::unique_ptr<agent_cpp::Callback>>{}, instructions);
	return local_state;
}

//===--------------------------------------------------------------------===//
// ai_enrich: execute
//===--------------------------------------------------------------------===//
// Runs one agent turn per row: row content in as the user message,
// grammar-constrained JSON out, parsed into the STRUCT's child vectors.
// Any field value that fails LooksGarbled() triggers a single whole-row
// retry at a nonzero temperature; only the fields that were actually bad
// are replaced from the retry (good primary values are kept as-is). A
// field that's still bad after the retry comes back NULL rather than
// aborting the whole query.
static void AIEnrichFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<AIEnrichBindData>();
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<AIEnrichLocalState>();

	auto &content_vector = args.data[0];
	UnifiedVectorFormat content_data;
	content_vector.ToUnifiedFormat(args.size(), content_data);
	auto content_values = UnifiedVectorFormat::GetData<string_t>(content_data);

	auto &child_entries = StructVector::GetEntries(result);
	D_ASSERT(child_entries.size() == info.field_names.size());
	for (auto &entry : child_entries) {
		entry->SetVectorType(VectorType::FLAT_VECTOR);
	}

	for (idx_t row = 0; row < args.size(); row++) {
		auto content_idx = content_data.sel->get_index(row);
		if (!content_data.validity.RowIsValid(content_idx)) {
			for (auto &entry : child_entries) {
				FlatVector::SetNull(*entry, row, true);
			}
			continue;
		}
		string row_content = content_values[content_idx].GetString();

		std::unordered_map<string, string> primary;
		bool primary_ok = GenerateEnrichmentFields(*lstate.agent, row_content, primary);

		bool any_field_needs_retry = false;
		for (auto &field : info.field_names) {
			auto it = primary_ok ? primary.find(field) : primary.end();
			if (it == primary.end() || LooksGarbled(it->second)) {
				any_field_needs_retry = true;
				break;
			}
		}

		std::unordered_map<string, string> retry;
		if (any_field_needs_retry) {
			GenerateEnrichmentFields(*lstate.retry_agent, row_content, retry);
			// Result (success or failure) is checked per-field below; a
			// failed/garbled retry for a given field just leaves it NULL.
		}

		for (idx_t child_idx = 0; child_idx < child_entries.size(); child_idx++) {
			auto &child_vector = *child_entries[child_idx];
			auto child_data = FlatVector::GetData<string_t>(child_vector);
			const auto &field = info.field_names[child_idx];

			auto primary_it = primary_ok ? primary.find(field) : primary.end();
			bool primary_good = primary_it != primary.end() && !LooksGarbled(primary_it->second);

			if (primary_good) {
				child_data[row] = StringVector::AddString(child_vector, primary_it->second);
				continue;
			}

			auto retry_it = retry.find(field);
			bool retry_good = retry_it != retry.end() && !LooksGarbled(retry_it->second);
			if (retry_good) {
				child_data[row] = StringVector::AddString(child_vector, retry_it->second);
			} else {
				FlatVector::SetNull(child_vector, row, true);
			}
		}
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
}

static void LoadInternal(ExtensionLoader &loader) {
	// ai_enrich(content VARCHAR, schema VARCHAR) -> STRUCT(...)
	// v1: simple flat-string schema only, no options, no knowledge_sources.
	ScalarFunction ai_enrich_fun("ai_enrich", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalTypeId::STRUCT,
	                             AIEnrichFun, AIEnrichBind, /*bind_extended=*/nullptr, /*statistics=*/nullptr,
	                             AIEnrichInitLocalState);
	ai_enrich_fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(ai_enrich_fun);
}

void DuckagentExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string DuckagentExtension::Name() {
	return "duckagent";
}

std::string DuckagentExtension::Version() const {
#ifdef EXT_VERSION_DUCKAGENT
	return EXT_VERSION_DUCKAGENT;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckagent, loader) {
	duckdb::LoadInternal(loader);
}
}
