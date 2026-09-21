#define DUCKDB_EXTENSION_MAIN

#include "duckagent_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/main/connection.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "yyjson.hpp"

// agent.cpp - local LLM agent runtime (see CMakeLists.txt FetchContent block)
#include "agent.h"
#include "error.h"
#include "model.h"
#include "llama.h"

#include <cstdlib>
#include <mutex>
#include <thread>
#include <algorithm>

namespace duckdb {

using namespace duckdb_yyjson;

//===--------------------------------------------------------------------===//
// ai_enrich: bind data
//===--------------------------------------------------------------------===//
// Parsed field names for the v1 flat VARCHAR schema.
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

// Parse a flat JSON array of field names.
static vector<string> ParseSimpleSchemaFields(const string &schema_json) {
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(schema_json.data(), schema_json.size(), 0),
	                                                       yyjson_doc_free);
	auto root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
	if (!root || !yyjson_is_arr(root)) {
		throw InvalidInputException("ai_enrich: schema must be a JSON array of field names");
	}
	if (yyjson_arr_size(root) == 0) {
		throw InvalidInputException("ai_enrich: schema array must contain at least one field name");
	}

	vector<string> fields;
	fields.reserve(yyjson_arr_size(root));
	size_t index, max;
	yyjson_val *value;
	yyjson_arr_foreach(root, index, max, value) {
		if (!yyjson_is_str(value) || yyjson_get_len(value) == 0) {
			throw InvalidInputException("ai_enrich: schema fields must be non-empty strings");
		}
		fields.emplace_back(yyjson_get_str(value), yyjson_get_len(value));
	}

	return fields;
}

//===--------------------------------------------------------------------===//
// ai_enrich: bind
//===--------------------------------------------------------------------===//
// Resolves the STRUCT return type from the constant schema argument.
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
// Silence llama.cpp's process-wide diagnostic output.
static void SilentGgmlLogCallback(ggml_log_level /*level*/, const char * /*text*/, void * /*user_data*/) {
}

// Share immutable weights across calls; each worker owns its model context.
// TODO: replace DUCKAGENT_MODEL_PATH with an extension setting.
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
// Builds a flat JSON grammar in schema order. The value-length cap prevents
// grammar-valid but runaway generation.
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

// Maximum characters in one generated field value.
static constexpr idx_t kEnrichMaxFieldValueLength = 120;

static string BuildEnrichGrammar(const vector<string> &field_names) {
	string root = "root ::= \"{\" ws";
	for (idx_t i = 0; i < field_names.size(); i++) {
		if (i > 0) {
			root += " \",\" ws";
		}
		root += " \"\\\"" + EscapeGbnfLiteral(field_names[i]) + "\\\":\" ws string";
	}
	root += " \"}\" ws\n\n";

	string grammar = root;
	grammar += "string ::=\n";
	grammar += "  \"\\\"\" (\n";
	grammar += "    [^\"\\\\\\x7F\\x00-\\x1F] |\n";
	grammar += "    \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n";
	grammar += "  ){0," + std::to_string(kEnrichMaxFieldValueLength) + "} \"\\\"\" ws\n";
	grammar += "\n";
	grammar += "ws ::= | \" \" | \"\\n\" [ \\t]{0,20}\n";
	return grammar;
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

// Parses the grammar-constrained response into schema order.
static bool TryParseEnrichmentFields(const string &json, const vector<string> &field_names, vector<string> &out) {
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(json.data(), json.size(), 0), yyjson_doc_free);
	auto root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
	if (!root || !yyjson_is_obj(root) || yyjson_obj_size(root) != field_names.size()) {
		return false;
	}

	out.clear();
	out.reserve(field_names.size());
	for (const auto &field : field_names) {
		auto value = yyjson_obj_getn(root, field.data(), field.size());
		if (!yyjson_is_str(value)) {
			return false;
		}
		out.emplace_back(yyjson_get_str(value), yyjson_get_len(value));
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Sanity check for generated field values
//===--------------------------------------------------------------------===//
// Reject common grammar-valid failure modes from small local models.
static bool LooksGarbled(const string &value) {
	if (value.empty()) {
		return true;
	}
	// Defense-in-depth alongside the grammar-level length cap
	// (kEnrichMaxFieldValueLength): a genuine short factual field value
	// should never be a multi-sentence essay.
	if (value.size() > kEnrichMaxFieldValueLength) {
		return true;
	}
	for (char c : value) {
		if (c == '<' || c == '>' || c == '{' || c == '}' || c == '[' || c == ']') {
			return true;
		}
	}
	if (value.rfind("http://", 0) == 0 || value.rfind("https://", 0) == 0) {
		return true;
	}
	if (value[0] == '/') {
		return true;
	}
	return false;
}

//===--------------------------------------------------------------------===//
// Runs one agent turn and parses its flat JSON response.
//===--------------------------------------------------------------------===//
static bool GenerateEnrichmentFields(agent_cpp::Agent &agent, const string &content,
                                      const vector<string> &field_names, vector<string> &out) {
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
	return TryParseEnrichmentFields(response_text, field_names, out);
}

//===--------------------------------------------------------------------===//
// Minor formatting cleanup for otherwise-good generated values
//===--------------------------------------------------------------------===//
// Remove leading whitespace and a cosmetic '+' from generated values.
static void NormalizeFieldValue(string &value) {
	idx_t start = 0;
	while (start < value.size() &&
	       (value[start] == '+' || isspace(static_cast<unsigned char>(value[start])))) {
		start++;
	}
	if (start > 0) {
		value.erase(0, start);
	}
}

static void NormalizeFields(vector<string> &fields) {
	for (auto &field : fields) {
		NormalizeFieldValue(field);
	}
}


// Reuse schema-specific agents for every chunk handled by this worker.
// Retry rejected values with sampling enabled to escape greedy failures.
static constexpr float kEnrichRetryTemperature = 0.7F;

struct AIEnrichLocalState : public FunctionLocalState {
	std::shared_ptr<agent_cpp::Agent> agent;
	std::shared_ptr<agent_cpp::Agent> retry_agent;
};

static unique_ptr<FunctionLocalState> AIEnrichInitLocalState(ExpressionState &state,
                                                              const BoundFunctionExpression &/*expr*/,
                                                              FunctionData *bind_data_p) {
	auto &info = bind_data_p->Cast<AIEnrichBindData>();

	auto weights = GetSharedModelWeights();
	auto grammar = BuildEnrichGrammar(info.field_names);
	string instructions = BuildEnrichInstructions(info.field_names);

	// Divide cores across DuckDB workers to avoid oversubscribing the host.
	auto num_duckdb_threads =
	    static_cast<idx_t>(std::max<int32_t>(1, TaskScheduler::GetScheduler(state.GetContext()).NumberOfThreads()));
	auto hw_threads = std::max<unsigned>(1U, std::thread::hardware_concurrency());
	auto threads_per_context = std::max<idx_t>(1, hw_threads / num_duckdb_threads);

	agent_cpp::ModelConfig primary_config;
	primary_config.grammar = grammar;
	primary_config.grammar_root = "root";
	primary_config.temp = 0.0F; // deterministic: field extraction, not creative writing
	primary_config.n_threads = static_cast<int>(threads_per_context);
	primary_config.n_threads_batch = static_cast<int>(threads_per_context);

	agent_cpp::ModelConfig retry_config;
	retry_config.grammar = grammar;
	retry_config.grammar_root = "root";
	retry_config.temp = kEnrichRetryTemperature;
	retry_config.n_threads = static_cast<int>(threads_per_context);
	retry_config.n_threads_batch = static_cast<int>(threads_per_context);

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
// Enrich each input row. Rejected primary values get one sampled retry;
// values still rejected become NULL without failing the query.
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

		vector<string> primary;
		bool primary_ok = GenerateEnrichmentFields(*lstate.agent, row_content, info.field_names, primary);
		if (primary_ok) {
			NormalizeFields(primary);
		}

		bool needs_retry = !primary_ok;
		for (const auto &field : primary) {
			needs_retry = needs_retry || LooksGarbled(field);
		}

		vector<string> retry;
		bool retry_ok = false;
		if (needs_retry) {
			retry_ok = GenerateEnrichmentFields(*lstate.retry_agent, row_content, info.field_names, retry);
			if (retry_ok) {
				NormalizeFields(retry);
			}
		}

		for (idx_t child_idx = 0; child_idx < child_entries.size(); child_idx++) {
			auto &child_vector = *child_entries[child_idx];
			auto child_data = FlatVector::GetData<string_t>(child_vector);
			bool primary_good = primary_ok && !LooksGarbled(primary[child_idx]);

			if (primary_good) {
				child_data[row] = StringVector::AddString(child_vector, primary[child_idx]);
				continue;
			}

			bool retry_good = retry_ok && !LooksGarbled(retry[child_idx]);
			if (retry_good) {
				child_data[row] = StringVector::AddString(child_vector, retry[child_idx]);
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

	//===----------------------------------------------------------------===//
	// ai_enrich_distinct: enrich one representative row per distinct key.
	//===----------------------------------------------------------------===//
	// `content_expr` may reference any source column. Join the compact result
	// back to the source through ai_enrich_distinct_key.
	static const char *kEnrichDistinctMacroSql = R"SQL(
CREATE OR REPLACE MACRO ai_enrich_distinct(source, key_column, content_expr, schema) AS TABLE
SELECT key_column AS ai_enrich_distinct_key,
       ai_enrich(content_expr, schema) AS enrichment
FROM (
    SELECT DISTINCT ON (key_column) *
    FROM query_table(source::VARCHAR)
) AS ai_enrich_distinct_rows;
)SQL";

	Connection conn(loader.GetDatabaseInstance());
	auto macro_result = conn.Query(kEnrichDistinctMacroSql);
	if (macro_result->HasError()) {
		throw IOException("duckagent: failed to register ai_enrich_distinct macro: %s", macro_result->GetError());
	}
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
