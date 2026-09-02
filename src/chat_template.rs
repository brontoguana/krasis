//! Chat template engine — applies Jinja2 chat templates in Rust.
//!
//! Uses minijinja to render HuggingFace-style chat templates. Replaces
//! the Python `tokenizer.apply_chat_template()` call so that the entire
//! request path (tokenize → prefill → decode → detokenize) can happen
//! without Python in the per-request hot path except for prefill.

use minijinja::value::{Kwargs, ValueKind};
use serde_json;

/// Native tool-call grammar declared by the loaded chat template.
///
/// This is derived from the template contract rather than the model name so a
/// checkpoint revision cannot silently inherit a parser for a different
/// grammar. `Unsupported` means the template does not declare an output form
/// that Krasis can safely translate to OpenAI structured tool calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolCallFormat {
    Unsupported,
    QwenJson,
    FunctionXml,
    GlmXml,
    DeepseekDsml,
    Gemma,
    Minimax,
}

impl ToolCallFormat {
    pub fn start_marker(self) -> Option<&'static str> {
        match self {
            Self::Unsupported => None,
            Self::QwenJson | Self::FunctionXml | Self::GlmXml => Some("<tool_call>"),
            Self::DeepseekDsml => Some("<｜DSML｜tool_calls>"),
            Self::Gemma => Some("<|tool_call>"),
            Self::Minimax => Some("<minimax:tool_call>"),
        }
    }

    /// Complete grammar tokens which may be registered as tokenizer special
    /// tokens and therefore need selective preservation during tool requests.
    pub fn preserved_special_tokens(self) -> &'static [&'static str] {
        match self {
            Self::Unsupported => &[],
            Self::QwenJson | Self::FunctionXml | Self::GlmXml => &["<tool_call>", "</tool_call>"],
            Self::DeepseekDsml => &["<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"],
            Self::Gemma => &["<|tool_call>", "<tool_call|>", "<|\"|>"],
            Self::Minimax => &["<minimax:tool_call>", "</minimax:tool_call>"],
        }
    }
}

/// A Jinja2 chat template engine for HuggingFace models.
pub struct ChatTemplateEngine {
    template_source: String,
    bos_token: String,
    eos_token: String,
    tool_call_format: ToolCallFormat,
    disabled_thinking_scaffold_history_stable: bool,
}

// DeepSeek-V2/V2-Lite chat format (plain text User:/Assistant:)
const DEEPSEEK_CHAT_TEMPLATE: &str = concat!(
    "{% if not add_generation_prompt is defined %}",
    "{% set add_generation_prompt = false %}{% endif %}",
    "{{ bos_token }}",
    "{% for message in messages %}",
    "{% if message['role'] == 'user' %}",
    "{{ 'User: ' + message['content'] + '\n\n' }}",
    "{% elif message['role'] == 'assistant' %}",
    "{{ 'Assistant: ' + message['content'] + eos_token }}",
    "{% elif message['role'] == 'system' %}",
    "{{ message['content'] + '\n\n' }}",
    "{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "{{ 'Assistant:' }}",
    "{% endif %}",
);

const DEEPSEEK_V4_CHAT_TEMPLATE: &str =
    include_str!("../python/krasis/chat_templates/deepseek_v4.jinja");

impl ChatTemplateEngine {
    /// Load a chat template from tokenizer_config.json.
    ///
    /// Reads the `chat_template` field from the config file.
    /// Falls back to DeepSeek format if no template is found.
    pub fn from_config(tokenizer_config_path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(tokenizer_config_path)
            .map_err(|e| format!("Failed to read {}: {}", tokenizer_config_path, e))?;
        let config: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        // Extract chat_template — can be a string or a list of {name, template} objects.
        // Some model snapshots keep tokenizer_config.json at chat_template=null
        // and ship the real template beside it as chat_template.jinja.
        let template_source = if let Some(ct) = config.get("chat_template") {
            match ct {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Array(arr) => {
                    // List of templates — prefer "default" or first one
                    let mut default_tmpl = None;
                    let mut first_tmpl = None;
                    for item in arr {
                        if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                            if let Some(tmpl) = item.get("template").and_then(|t| t.as_str()) {
                                if first_tmpl.is_none() {
                                    first_tmpl = Some(tmpl.to_string());
                                }
                                if name == "default" {
                                    default_tmpl = Some(tmpl.to_string());
                                }
                            }
                        }
                    }
                    match default_tmpl.or(first_tmpl) {
                        Some(template) => template,
                        None => resolve_missing_chat_template(tokenizer_config_path)?,
                    }
                }
                _ => resolve_missing_chat_template(tokenizer_config_path)?,
            }
        } else {
            resolve_missing_chat_template(tokenizer_config_path)?
        };

        let (template_source, disabled_thinking_scaffold_history_stable) =
            make_disabled_thinking_scaffold_history_stable(template_source);

        // Extract bos_token and eos_token
        let bos_token = extract_token(&config, "bos_token").unwrap_or_default();
        let eos_token = extract_token(&config, "eos_token").unwrap_or_default();

        let tool_call_format = detect_tool_call_format(&template_source);
        log::info!(
            "ChatTemplateEngine: loaded template ({} chars), bos={:?}, eos={:?}, tool_call_format={:?}, disabled_thinking_scaffold_history_stable={}",
            template_source.len(),
            bos_token,
            eos_token,
            tool_call_format,
            disabled_thinking_scaffold_history_stable,
        );

        Ok(ChatTemplateEngine {
            template_source,
            bos_token,
            eos_token,
            tool_call_format,
            disabled_thinking_scaffold_history_stable,
        })
    }

    pub fn tool_call_format(&self) -> ToolCallFormat {
        self.tool_call_format
    }

    pub fn compatibility_source(&self) -> &str {
        &self.template_source
    }

    /// True when a thinking-disabled assistant generation suffix is rendered
    /// identically when that assistant response later appears in history.
    /// This makes the ordinary full prompt a future exact-token prefix and
    /// permits terminal capture without a synthetic mid-prefill split.
    pub fn disabled_thinking_generation_prompt_is_history_stable(&self) -> bool {
        self.disabled_thinking_scaffold_history_stable
    }

    /// Apply the chat template to a list of messages.
    ///
    /// `messages_json` is a JSON array of {role, content} objects.
    /// `tools_json` is an optional JSON array of tool definitions (OpenAI format).
    /// Returns the rendered text string ready for tokenization.
    pub fn apply(
        &self,
        messages_json: &str,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        self.apply_with_tools(messages_json, "", add_generation_prompt, enable_thinking)
    }

    /// Apply with optional tools array for accurate token estimation.
    pub fn apply_with_tools(
        &self,
        messages_json: &str,
        tools_json: &str,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        self.apply_with_tools_inner(
            messages_json,
            tools_json,
            add_generation_prompt,
            enable_thinking,
            false,
        )
    }

    /// Apply a multimodal-capable chat template without flattening image parts.
    pub fn apply_multimodal_with_tools(
        &self,
        messages_json: &str,
        tools_json: &str,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        self.apply_with_tools_inner(
            messages_json,
            tools_json,
            add_generation_prompt,
            enable_thinking,
            true,
        )
    }

    fn apply_with_tools_inner(
        &self,
        messages_json: &str,
        tools_json: &str,
        add_generation_prompt: bool,
        enable_thinking: bool,
        preserve_multimodal_content: bool,
    ) -> Result<String, String> {
        let mut messages: serde_json::Value = serde_json::from_str(messages_json)
            .map_err(|e| format!("Failed to parse messages JSON: {}", e))?;
        let tools: serde_json::Value = if tools_json.is_empty() {
            serde_json::Value::Array(vec![])
        } else {
            serde_json::from_str(tools_json)
                .map_err(|e| format!("Failed to parse tools JSON: {}", e))?
        };

        // Pre-process messages before rendering. Keep plain string content unchanged.
        // OpenAI text content parts are flattened to the string content expected by
        // current text-only HF templates; non-text parts fail instead of rendering
        // an array/object into the prompt.
        // OpenAI format sends arguments as a JSON string, but Jinja templates
        // use `arguments|items` which requires a dict/mapping.
        if let Some(msgs) = messages.as_array_mut() {
            for msg in msgs.iter_mut() {
                if preserve_multimodal_content {
                    normalize_multimodal_content_parts_for_templates(msg)?;
                } else {
                    normalize_text_content_parts(msg)?;
                }
                if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                    for tc in tool_calls.iter_mut() {
                        // Determine which object holds arguments: tc.function or tc itself
                        let has_function = tc.get("function").is_some();
                        let fn_obj = if has_function {
                            tc.get_mut("function").unwrap()
                        } else {
                            &mut *tc
                        };
                        if let Some(args_str) = fn_obj
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .map(String::from)
                        {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&args_str)
                            {
                                fn_obj["arguments"] = parsed;
                            }
                        }
                    }
                }
            }
        }

        let mut env = minijinja::Environment::new();

        // Hugging Face compiles every chat template with these two Jinja
        // whitespace options. Matching them globally is part of the template
        // contract: otherwise indented control blocks add prompt tokens that
        // are absent from tokenizer.apply_chat_template().
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        // MiniJinja renders booleans and None with Rust spellings, while HF's
        // Jinja environment uses Python spellings. Some shipped templates use
        // `|string` while serializing tool schemas, so provide Python/Jinja
        // semantics for the complete JSON-shaped value domain.
        env.add_filter("string", python_string_filter);

        // Add tojson filter (used by Qwen templates to serialize tool parameters)
        env.add_filter("tojson", to_json_filter);

        // Shipped templates accept OpenAI tool-call arguments as either an
        // object or a JSON string. Keep malformed JSON visible instead of
        // silently rendering an invalid tool invocation.
        env.add_filter("from_json", from_json_filter);
        // Some pinned Hugging Face templates use the spelling without an
        // underscore. Both names are explicit template contracts, not a
        // fallback from one rendering path to another.
        env.add_filter("fromjson", from_json_filter);

        // Add raise_exception function (used by some templates)
        env.add_function("raise_exception", raise_exception);

        // Add strftime_now function (used by some templates)
        env.add_function("strftime_now", strftime_now);

        // Handle Python string methods used by HuggingFace templates
        env.set_unknown_method_callback(|_state, value, method, args| match method {
            "startswith" => {
                let s = value.as_str().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "startswith requires a string",
                    )
                })?;
                let prefix = args.first().and_then(|a| a.as_str()).ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "startswith requires a string argument",
                    )
                })?;
                Ok(minijinja::Value::from(s.starts_with(prefix)))
            }
            "endswith" => {
                let s = value.as_str().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "endswith requires a string",
                    )
                })?;
                let suffix = args.first().and_then(|a| a.as_str()).ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "endswith requires a string argument",
                    )
                })?;
                Ok(minijinja::Value::from(s.ends_with(suffix)))
            }
            "strip" => {
                let s = value.as_str().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "strip requires a string",
                    )
                })?;
                Ok(minijinja::Value::from(s.trim()))
            }
            "lstrip" => {
                let s = value.as_str().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "lstrip requires a string",
                    )
                })?;
                let chars = args.first().and_then(|a| a.as_str());
                Ok(minijinja::Value::from(match chars {
                    Some(c) => s.trim_start_matches(|ch: char| c.contains(ch)),
                    None => s.trim_start(),
                }))
            }
            "rstrip" => {
                let s = value.as_str().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "rstrip requires a string",
                    )
                })?;
                let chars = args.first().and_then(|a| a.as_str());
                Ok(minijinja::Value::from(match chars {
                    Some(c) => s.trim_end_matches(|ch: char| c.contains(ch)),
                    None => s.trim_end(),
                }))
            }
            "split" => {
                let s = value.as_str().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "split requires a string",
                    )
                })?;
                let sep = args.first().and_then(|a| a.as_str());
                let parts: Vec<minijinja::Value> = match sep {
                    Some(sep) => s.split(sep).map(minijinja::Value::from).collect(),
                    None => s.split_whitespace().map(minijinja::Value::from).collect(),
                };
                Ok(minijinja::Value::from(parts))
            }
            "get" => {
                let key = args.first().ok_or_else(|| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "get requires a key argument",
                    )
                })?;
                match value.get_item(key) {
                    Ok(item) if !item.is_undefined() => Ok(item),
                    Ok(_) => Ok(args.get(1).cloned().unwrap_or(minijinja::Value::UNDEFINED)),
                    Err(_) => Ok(args.get(1).cloned().unwrap_or(minijinja::Value::UNDEFINED)),
                }
            }
            "items" => {
                if value.kind() != minijinja::value::ValueKind::Map {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "items requires a mapping",
                    ));
                }
                if !args.is_empty() {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "items takes no arguments",
                    ));
                }
                let keys = value.try_iter().map_err(|error| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        format!("items could not iterate mapping: {}", error),
                    )
                })?;
                let mut items = Vec::new();
                for key in keys {
                    let item = value.get_item(&key).map_err(|error| {
                        minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            format!("items could not read mapping value: {}", error),
                        )
                    })?;
                    items.push(minijinja::Value::from(vec![key, item]));
                }
                Ok(minijinja::Value::from(items))
            }
            _ => Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("unknown method: {}", method),
            )),
        });

        // Register after filters/functions so templates that reference them
        // compile against the complete environment.
        env.add_template("chat", &self.template_source)
            .map_err(|e| format!("Failed to compile chat template: {}", e))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| format!("Failed to get template: {}", e))?;

        let ctx = minijinja::context! {
            messages => messages,
            tools => tools,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
            preserve_thinking => false,
        };

        let rendered = tmpl
            .render(ctx)
            .map_err(|e| format!("Template render failed: {}", e))?;
        let rendered = rendered.trim_start_matches(['\n', '\r']).to_string();
        Ok(close_initial_think_block_if_disabled(
            rendered,
            add_generation_prompt,
            enable_thinking,
        ))
    }
}

/// Qwen-derived templates intentionally discard old reasoning, but when
/// thinking is disabled Krasis has already seeded an *empty* closed thinking
/// block in the generation prompt. Dropping those empty marker tokens on the
/// next render destroys exact prefix reuse despite carrying no reasoning.
///
/// Extend only the template's existing history condition, and only for empty
/// reasoning in thinking-disabled mode. Non-empty reasoning and thinking-on
/// behavior remain exactly as declared by the checkpoint. Matching is on the
/// template contract, never a model name.
fn make_disabled_thinking_scaffold_history_stable(template: String) -> (String, bool) {
    const HISTORY_CONDITION: &str = "{%- if loop.index0 > ns.last_query_index %}";
    const STABLE_HISTORY_CONDITION: &str = "{%- if loop.index0 > ns.last_query_index or (enable_thinking is defined and enable_thinking is false and not reasoning_content) %}";
    const DISABLED_SCAFFOLD: &str = "<think>\\n\\n</think>\\n\\n";
    if template.contains(HISTORY_CONDITION) && template.contains(DISABLED_SCAFFOLD) {
        (
            template.replace(HISTORY_CONDITION, STABLE_HISTORY_CONDITION),
            true,
        )
    } else {
        (template, false)
    }
}

fn detect_tool_call_format(template: &str) -> ToolCallFormat {
    if template.contains("｜DSML｜") && template.contains("invoke name=") {
        ToolCallFormat::DeepseekDsml
    } else if template.contains("<|tool_call>call:") {
        ToolCallFormat::Gemma
    } else if template.contains("<minimax:tool_call>") && template.contains("<invoke name=\"") {
        ToolCallFormat::Minimax
    } else if template.contains("<arg_key>") && template.contains("<arg_value>") {
        ToolCallFormat::GlmXml
    } else if template.contains("{\"name\": <function-name>")
        || template.contains("{\\\"name\\\": <function-name>")
    {
        ToolCallFormat::QwenJson
    } else if template.contains("<function=example_function_name>")
        || (template.contains("<function=") && template.contains("<parameter="))
    {
        ToolCallFormat::FunctionXml
    } else {
        ToolCallFormat::Unsupported
    }
}

fn close_initial_think_block_if_disabled(
    mut rendered: String,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    if add_generation_prompt && !enable_thinking {
        const QWEN_OPEN_THINK_SUFFIX: &str = "<|im_start|>assistant\n<think>\n";
        if rendered.ends_with(QWEN_OPEN_THINK_SUFFIX) {
            rendered.push_str("\n</think>\n\n");
        } else if rendered.ends_with("<think>") {
            // Some checkpoint templates unconditionally emit a bare opening
            // marker in their generation prompt and expose no
            // `enable_thinking` branch.  Close only that exact terminal marker
            // so disabled mode remains template-driven rather than
            // model-name-driven.
            rendered.push_str("</think>");
        }
    }
    rendered
}

fn load_sibling_chat_template(tokenizer_config_path: &str) -> Option<String> {
    let config_path = std::path::Path::new(tokenizer_config_path);
    let template_path = config_path.parent()?.join("chat_template.jinja");
    match std::fs::read_to_string(&template_path) {
        Ok(template) => {
            log::info!("Loaded chat template from {}", template_path.display());
            Some(template)
        }
        Err(_) => None,
    }
}

fn resolve_missing_chat_template(tokenizer_config_path: &str) -> Result<String, String> {
    if let Some(template) = load_sibling_chat_template(tokenizer_config_path) {
        return Ok(template);
    }

    let config_path = std::path::Path::new(tokenizer_config_path);
    let model_config_path = config_path
        .parent()
        .unwrap_or(config_path)
        .join("config.json");
    match std::fs::read_to_string(&model_config_path) {
        Ok(data) => {
            let config: serde_json::Value = serde_json::from_str(&data).map_err(|e| {
                format!(
                    "Failed to parse model config {} while resolving chat template: {}",
                    model_config_path.display(),
                    e
                )
            })?;
            let model_type = config
                .get("model_type")
                .and_then(|value| value.as_str())
                .or_else(|| {
                    config
                        .get("text_config")
                        .and_then(|value| value.get("model_type"))
                        .and_then(|value| value.as_str())
                });
            if model_type == Some("deepseek_v4") {
                log::info!(
                    "Checkpoint ships no Jinja template; using bundled DeepSeek-V4 template"
                );
                return Ok(DEEPSEEK_V4_CHAT_TEMPLATE
                    .trim_end_matches(['\r', '\n'])
                    .to_string());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "Failed to read model config {} while resolving chat template: {}",
                model_config_path.display(),
                error
            ));
        }
    }

    log::info!("No chat_template in config — using DeepSeek format fallback");
    Ok(DEEPSEEK_CHAT_TEMPLATE.to_string())
}

/// Extract a token string from tokenizer_config.json.
/// Handles both `"bos_token": "<s>"` and `"bos_token": {"content": "<s>", ...}`.
fn extract_token(config: &serde_json::Value, key: &str) -> Option<String> {
    match config.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(obj) => obj
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from),
        _ => None,
    }
}

fn normalize_text_content_parts(message: &mut serde_json::Value) -> Result<(), String> {
    let Some(content) = message.get_mut("content") else {
        return Ok(());
    };
    let serde_json::Value::Array(parts) = content else {
        return Ok(());
    };

    let mut text = String::new();
    for (idx, part) in parts.iter().enumerate() {
        let obj = part
            .as_object()
            .ok_or_else(|| format!("structured content part {} must be an object", idx))?;
        let part_type = obj.get("type").and_then(|v| v.as_str());
        if part_type == Some("text") || obj.contains_key("text") {
            let part_text = obj
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("structured content part {} text must be a string", idx))?;
            text.push_str(part_text);
            continue;
        }
        return Err(format!(
            "unsupported non-text structured content part {}",
            idx
        ));
    }
    *content = serde_json::Value::String(text);
    Ok(())
}

fn normalize_multimodal_content_parts_for_templates(
    message: &mut serde_json::Value,
) -> Result<(), String> {
    let Some(content) = message.get_mut("content") else {
        return Ok(());
    };
    let serde_json::Value::Array(parts) = content else {
        return Ok(());
    };

    for (idx, part) in parts.iter_mut().enumerate() {
        let obj = part
            .as_object_mut()
            .ok_or_else(|| format!("structured content part {} must be an object", idx))?;
        let part_type = obj.get("type").and_then(|v| v.as_str());
        let is_image_like = obj.contains_key("image")
            || obj.contains_key("image_url")
            || matches!(part_type, Some("image" | "image_url" | "input_image"));
        if !is_image_like {
            continue;
        }

        if !obj.contains_key("image") {
            if let Some(image_url) = obj.get("image_url").cloned() {
                obj.insert("image".to_string(), image_url);
            }
        }
        obj.insert(
            "type".to_string(),
            serde_json::Value::String("image".to_string()),
        );
    }
    Ok(())
}

/// raise_exception function for Jinja2 templates.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// Hugging Face templates pass Jinja2's `ensure_ascii` keyword to `tojson`.
/// MiniJinja's built-in filter does not accept that keyword, so keep Krasis's
/// serde conversion while implementing the declared call signature.
fn to_json_filter(
    value: minijinja::Value,
    positional_indent: Option<minijinja::Value>,
    kwargs: Kwargs,
) -> Result<String, minijinja::Error> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let keyword_indent = kwargs.get::<Option<minijinja::Value>>("indent")?;
    kwargs.assert_all_used()?;

    if positional_indent.is_some() && keyword_indent.is_some() {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::TooManyArguments,
            "tojson indent was supplied both positionally and by keyword",
        ));
    }

    let indent = positional_indent.or(keyword_indent);
    let json_value = minijinja_value_to_json(&value);
    let mut rendered = match indent {
        None => serde_json::to_string(&json_value),
        Some(indent) => {
            let width = if let Ok(enabled) = bool::try_from(indent.clone()) {
                if enabled {
                    2
                } else {
                    0
                }
            } else {
                usize::try_from(indent).map_err(|_| {
                    minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "tojson indent must be a boolean or non-negative integer",
                    )
                })?
            };
            if width == 0 {
                serde_json::to_string(&json_value)
            } else {
                let mut output = Vec::new();
                let indentation = " ".repeat(width);
                let formatter =
                    serde_json::ser::PrettyFormatter::with_indent(indentation.as_bytes());
                let mut serializer = serde_json::Serializer::with_formatter(&mut output, formatter);
                serde::Serialize::serialize(&json_value, &mut serializer)
                    .map(|_| String::from_utf8(output).expect("JSON serializer emits UTF-8"))
            }
        }
    }
    .map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("cannot serialize to JSON: {}", error),
        )
    })?;

    if ensure_ascii {
        let mut ascii = String::with_capacity(rendered.len());
        use std::fmt::Write as _;
        for character in rendered.chars() {
            if character.is_ascii() {
                ascii.push(character);
            } else {
                let mut encoded = [0u16; 2];
                for unit in character.encode_utf16(&mut encoded) {
                    write!(&mut ascii, "\\u{unit:04x}").expect("writing to String cannot fail");
                }
            }
        }
        rendered = ascii;
    }

    Ok(rendered)
}

fn from_json_filter(value: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    let input = value.as_str().ok_or_else(|| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "JSON parser filter requires a string",
        )
    })?;
    let parsed: serde_json::Value = serde_json::from_str(input).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("JSON parser filter received invalid JSON: {}", error),
        )
    })?;
    Ok(minijinja::Value::from_serialize(parsed))
}

/// Match Python/Jinja's `string` filter for JSON-shaped template values.
fn python_string_filter(value: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    if value.is_undefined() {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::UndefinedError,
            "cannot convert an undefined value to a string",
        ));
    }
    if value.kind() == ValueKind::String {
        return Ok(value);
    }
    Ok(minijinja::Value::from(python_value_repr(&value, false)))
}

fn python_value_repr(value: &minijinja::Value, nested: bool) -> String {
    match value.kind() {
        ValueKind::None => "None".to_string(),
        ValueKind::Bool => {
            if value.is_true() {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        ValueKind::String => {
            let text = value.as_str().unwrap_or_default();
            if nested {
                python_quote_string(text)
            } else {
                text.to_string()
            }
        }
        ValueKind::Seq => {
            let items = value
                .try_iter()
                .map(|iter| {
                    iter.map(|item| python_value_repr(&item, true))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let mut items = Vec::new();
            if let Ok(keys) = value.try_iter() {
                for key in keys {
                    if let Ok(item) = value.get_item(&key) {
                        items.push(format!(
                            "{}: {}",
                            python_value_repr(&key, true),
                            python_value_repr(&item, true)
                        ));
                    }
                }
            }
            format!("{{{}}}", items.join(", "))
        }
        _ => value.to_string(),
    }
}

fn python_quote_string(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut output = String::with_capacity(value.len() + 2);
    output.push(quote);
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\x08' => output.push_str("\\b"),
            '\x0c' => output.push_str("\\f"),
            character if character == quote => {
                output.push('\\');
                output.push(character);
            }
            character if character.is_control() => {
                use std::fmt::Write as _;
                write!(&mut output, "\\u{:04x}", character as u32)
                    .expect("writing to String cannot fail");
            }
            character => output.push(character),
        }
    }
    output.push(quote);
    output
}

/// strftime_now function for Jinja2 templates (returns current date/time).
fn strftime_now(fmt: String) -> String {
    // Simple implementation covering common format strings
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // For the typical use case (%Y-%m-%d), we do a basic calculation
    // Full strftime is overkill — most templates just use %Y-%m-%d or similar
    if fmt.contains("%Y") || fmt.contains("%m") || fmt.contains("%d") {
        // Days since epoch
        let days = now / 86400;
        let (year, month, day) = days_to_ymd(days as i64);
        fmt.replace("%Y", &format!("{:04}", year))
            .replace("%m", &format!("{:02}", month))
            .replace("%d", &format!("{:02}", day))
    } else {
        format!("{}", now)
    }
}

/// Convert a minijinja Value to a serde_json Value for JSON serialization.
fn minijinja_value_to_json(value: &minijinja::Value) -> serde_json::Value {
    if value.is_none() || value.is_undefined() {
        serde_json::Value::Null
    } else if let Some(b) = value.as_str() {
        serde_json::Value::String(b.to_string())
    } else if let Ok(b) = bool::try_from(value.clone()) {
        serde_json::Value::Bool(b)
    } else if let Ok(n) = i64::try_from(value.clone()) {
        serde_json::json!(n)
    } else if let Ok(n) = f64::try_from(value.clone()) {
        serde_json::json!(n)
    } else if value.kind() == minijinja::value::ValueKind::Seq {
        let items: Vec<serde_json::Value> = value
            .try_iter()
            .map(|iter| iter.map(|v| minijinja_value_to_json(&v)).collect())
            .unwrap_or_default();
        serde_json::Value::Array(items)
    } else if value.kind() == minijinja::value::ValueKind::Map {
        let mut map = serde_json::Map::new();
        if let Ok(keys) = value.try_iter() {
            for key in keys {
                let key_str = key.to_string();
                if let Ok(val) = value.get_item(&key) {
                    map.insert(key_str, minijinja_value_to_json(&val));
                }
            }
        }
        serde_json::Value::Object(map)
    } else {
        // Fallback: use the display representation
        serde_json::Value::String(value.to_string())
    }
}

/// Convert days since epoch to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::{detect_tool_call_format, ChatTemplateEngine, ToolCallFormat};
    use std::fs;

    fn write_tokenizer_config(template: &str) -> String {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "krasis_chat_template_test_{}_{}",
            std::process::id(),
            nonce
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer_config.json");
        let data = serde_json::json!({
            "chat_template": template,
            "bos_token": "<s>",
            "eos_token": "</s>"
        });
        fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn preserve_thinking_defaults_false() {
        let template = concat!(
            "{% for message in messages %}",
            "{% if message.role == 'assistant' %}",
            "{% if preserve_thinking is defined and preserve_thinking is true %}",
            "KEEP:{{ message.content }}",
            "{% else %}",
            "DROP:{{ message.content }}",
            "{% endif %}",
            "{% endif %}",
            "{% endfor %}"
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let messages = r#"[{"role":"assistant","content":"old reasoning"}]"#;
        let rendered = engine.apply(messages, false, false).unwrap();
        assert_eq!(rendered, "DROP:old reasoning");
    }

    #[test]
    fn disabled_empty_thinking_scaffold_is_an_exact_future_history_prefix() {
        let template = concat!(
            "{%- set ns = namespace(last_query_index=messages|length - 1) %}",
            "{%- for message in messages %}",
            "{%- set content = message.content %}",
            "{%- if message.role == 'user' %}",
            "{{- '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}",
            "{%- elif message.role == 'assistant' %}",
            "{%- set reasoning_content = '' %}",
            "{%- if '</think>' in content %}",
            "{%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1]|trim %}",
            "{%- set content = content.split('</think>')[-1]|trim %}",
            "{%- endif %}",
            "{%- if loop.index0 > ns.last_query_index %}",
            "{{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}",
            "{%- else %}",
            "{{- '<|im_start|>assistant\\n' + content }}",
            "{%- endif %}",
            "{{- '<|im_end|>\\n' }}",
            "{%- endif %}",
            "{%- endfor %}",
            "{%- if add_generation_prompt %}",
            "{{- '<|im_start|>assistant\\n' }}",
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "{{- '<think>\\n\\n</think>\\n\\n' }}",
            "{%- else %}{{- '<think>\\n' }}{%- endif %}",
            "{%- endif %}",
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        assert!(engine.disabled_thinking_generation_prompt_is_history_stable());

        let first = engine
            .apply(r#"[{"role":"user","content":"first"}]"#, true, false)
            .unwrap();
        let next = engine
            .apply(
                r#"[{"role":"user","content":"first"},{"role":"assistant","content":"answer"},{"role":"user","content":"next"}]"#,
                true,
                false,
            )
            .unwrap();
        assert!(next.starts_with(&(first + "answer<|im_end|>\n")));
    }

    #[test]
    fn matches_hugging_face_jinja_whitespace_options() {
        let config = write_tokenizer_config(
            "start\n  {% if messages %}\n    {{ messages[0].content }}\n  {% endif %}\nend",
        );
        let engine = ChatTemplateEngine::from_config(&config).unwrap();
        let rendered = engine
            .apply(r#"[{"role":"user","content":"value"}]"#, false, false)
            .unwrap();
        assert_eq!(rendered, "start\n    value\nend");
    }

    #[test]
    fn supports_hugging_face_loop_controls() {
        let config = write_tokenizer_config(concat!(
            "{% for message in messages %}",
            "{{ message.content }}",
            "{% if message.content == 'stop' %}{% break %}{% endif %}",
            "{% endfor %}",
        ));
        let engine = ChatTemplateEngine::from_config(&config).unwrap();
        let rendered = engine
            .apply(
                r#"[{"role":"user","content":"first"},{"role":"user","content":"stop"},{"role":"user","content":"ignored"}]"#,
                false,
                false,
            )
            .unwrap();
        assert_eq!(rendered, "firststop");
    }

    #[test]
    fn string_filter_matches_python_for_json_shaped_values() {
        let config = write_tokenizer_config(
            "{{ values.false|string }}|{{ values.true|string }}|{{ values.none|string }}|{{ values.number|string }}|{{ values.text|string }}|{{ values.sequence|string }}|{{ values.mapping|string }}",
        );
        let engine = ChatTemplateEngine::from_config(&config).unwrap();
        let rendered = engine
            .apply(r#"[{"role":"user","content":"unused"}]"#, false, false)
            .unwrap_err();
        assert!(rendered.contains("undefined"));

        let config = write_tokenizer_config(
            "{{ tools[0].function.parameters.properties.false.default|string }}|{{ tools[0].function.parameters.properties.true.default|string }}|{{ tools[0].function.parameters.properties.none.default|string }}|{{ tools[0].function.parameters.properties.number.default|string }}|{{ tools[0].function.parameters.properties.text.default|string }}|{{ tools[0].function.parameters.properties.sequence.default|string }}|{{ tools[0].function.parameters.properties.mapping.default|string }}",
        );
        let engine = ChatTemplateEngine::from_config(&config).unwrap();
        let tools = r#"[{"type":"function","function":{"name":"probe","parameters":{"properties":{"false":{"default":false},"true":{"default":true},"none":{"default":null},"number":{"default":3.0},"text":{"default":"hello"},"sequence":{"default":[1,false]},"mapping":{"default":{"a":false,"b":null}}}}}}]"#;
        let rendered = engine.apply_with_tools("[]", tools, false, false).unwrap();
        assert_eq!(
            rendered,
            "False|True|None|3.0|hello|[1, False]|{'a': False, 'b': None}"
        );
    }

    #[test]
    fn enable_thinking_is_still_passed() {
        let template = concat!(
            "{% if add_generation_prompt %}",
            "{% if enable_thinking is defined and enable_thinking is false %}",
            "<think>\n\n</think>\n\n",
            "{% else %}",
            "<think>\n",
            "{% endif %}",
            "{% endif %}"
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        assert_eq!(
            engine
                .apply(r#"[{"role":"user","content":"hi"}]"#, true, false)
                .unwrap(),
            "<think>\n\n</think>\n\n"
        );
        assert_eq!(
            engine
                .apply(r#"[{"role":"user","content":"hi"}]"#, true, true)
                .unwrap(),
            "<think>\n"
        );
    }

    #[test]
    fn always_open_think_template_closes_when_disabled() {
        let template = concat!(
            "{{ bos_token }}",
            "{% for message in messages %}",
            "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}",
            "{% endfor %}",
            "{% if add_generation_prompt %}",
            "{{ '<|im_start|>assistant\n<think>\n' }}",
            "{% endif %}"
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        assert_eq!(
            engine
                .apply(r#"[{"role":"user","content":"hi"}]"#, true, false)
                .unwrap(),
            "<s><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            engine
                .apply(r#"[{"role":"user","content":"hi"}]"#, true, true)
                .unwrap(),
            "<s><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn text_content_parts_are_flattened_before_render() {
        let template = "{% for message in messages %}{{ message.content }}{% endfor %}";
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let messages =
            r#"[{"role":"user","content":[{"type":"text","text":"Hello"},{"text":" world"}]}]"#;
        let rendered = engine.apply(messages, false, false).unwrap();
        assert_eq!(rendered, "Hello world");
    }

    #[test]
    fn multimodal_image_url_parts_render_as_image_parts() {
        let template = concat!(
            "{% for message in messages %}",
            "{% for item in message.content %}",
            "{% if item.type == 'text' %}{{ item.text }}{% endif %}",
            "{% if item.type == 'image' %}<im_patch>{% endif %}",
            "{% endfor %}",
            "{% endfor %}"
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let messages = r#"[{"role":"user","content":[{"type":"text","text":"Look:"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]}]"#;
        let rendered = engine
            .apply_multimodal_with_tools(messages, "", false, false)
            .unwrap();
        assert_eq!(rendered, "Look:<im_patch>");
    }

    #[test]
    fn dict_get_method_matches_python_template_usage() {
        let template = concat!(
            "{% for message in messages %}",
            "{{ message.get('reasoning') or message.get('reasoning_content', '') }}",
            "{% endfor %}"
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let messages = r#"[{"role":"assistant","content":"x","reasoning_content":"thinking"}]"#;
        let rendered = engine.apply(messages, false, false).unwrap();
        assert_eq!(rendered, "thinking");
    }

    #[test]
    fn hf_tojson_keywords_and_fromjson_alias_render_tool_templates() {
        let template = concat!(
            "{% for tool in tools %}{{ tool | tojson(ensure_ascii=False) }}{% endfor %}",
            "{{ '{\"city\":\"München\",\"days\":2}' | fromjson | tojson(ensure_ascii=False) }}",
            "{{ tools[0] | tojson(ensure_ascii=True) }}",
        );
        let config_path = write_tokenizer_config(template);
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let rendered = engine
            .apply_with_tools(
                r#"[{"role":"user","content":"weather"}]"#,
                r#"[{"type":"function","function":{"name":"weather","description":"München weather","parameters":{"type":"object"}}}]"#,
                false,
                false,
            )
            .unwrap();
        assert!(rendered.contains("München weather"));
        assert!(rendered.contains(r#"{"city":"München","days":2}"#));
        assert!(rendered.contains(r#"M\u00fcnchen weather"#));
    }

    #[test]
    fn bundled_deepseek_v4_template_renders_chat_mode() {
        let config_path = write_tokenizer_config("{{ bos_token }}");
        let config_dir = std::path::Path::new(&config_path).parent().unwrap();
        fs::write(
            config_dir.join("config.json"),
            serde_json::json!({"model_type": "deepseek_v4"}).to_string(),
        )
        .unwrap();
        fs::write(
            &config_path,
            serde_json::json!({
                "chat_template": null,
                "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
                "eos_token": {"content": "<｜end▁of▁sentence｜>"}
            })
            .to_string(),
        )
        .unwrap();

        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let rendered = engine
            .apply(
                r#"[{"role":"user","content":"Compute 29 plus 34. Put the number first."}]"#,
                true,
                false,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>Compute 29 plus 34. Put the number first.<｜Assistant｜></think>"
        );
    }

    #[test]
    fn bundled_deepseek_v4_template_renders_image_placeholder_in_order() {
        let config_path = write_tokenizer_config("{{ bos_token }}");
        let config_dir = std::path::Path::new(&config_path).parent().unwrap();
        fs::write(
            config_dir.join("config.json"),
            serde_json::json!({"model_type": "deepseek_v4"}).to_string(),
        )
        .unwrap();
        fs::write(
            &config_path,
            serde_json::json!({
                "chat_template": null,
                "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
                "eos_token": {"content": "<｜end▁of▁sentence｜>"}
            })
            .to_string(),
        )
        .unwrap();
        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let rendered = engine
            .apply_multimodal_with_tools(
                r#"[{"role":"user","content":[{"type":"text","text":"Before "},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}},{"type":"text","text":" after"}]}]"#,
                "",
                true,
                false,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>Before <｜deepseek_image｜> after<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_v4_template_nests_tool_result_images_in_user_turns() {
        let config_path = write_tokenizer_config("{{ bos_token }}");
        let config_dir = std::path::Path::new(&config_path).parent().unwrap();
        fs::write(
            config_dir.join("config.json"),
            serde_json::json!({"model_type": "deepseek_v4"}).to_string(),
        )
        .unwrap();
        fs::write(
            &config_path,
            serde_json::json!({
                "chat_template": null,
                "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
                "eos_token": {"content": "<｜end▁of▁sentence｜>"}
            })
            .to_string(),
        )
        .unwrap();

        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let rendered = engine
            .apply_multimodal_with_tools(
                r#"[{"role":"assistant","content":"","tool_calls":[{"id":"call_vision","type":"function","function":{"name":"vision_analyze","arguments":"{\"image_url\":\"local.png\"}"}}]},{"role":"tool","tool_call_id":"call_vision","content":[{"type":"text","text":"Image loaded."},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc"}}]}]"#,
                "",
                true,
                false,
            )
            .unwrap();
        assert!(rendered
            .contains("<｜User｜><tool_result>Image loaded.<｜deepseek_image｜></tool_result>"));
        assert!(rendered.ends_with("<｜Assistant｜></think>"));
    }

    #[test]
    fn deepseek_v4_template_renders_openai_tool_calls() {
        let config_path = write_tokenizer_config("{{ bos_token }}");
        let config_dir = std::path::Path::new(&config_path).parent().unwrap();
        fs::write(
            config_dir.join("config.json"),
            serde_json::json!({"model_type": "deepseek_v4"}).to_string(),
        )
        .unwrap();
        fs::write(
            &config_path,
            serde_json::json!({
                "chat_template": null,
                "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
                "eos_token": {"content": "<｜end▁of▁sentence｜>"}
            })
            .to_string(),
        )
        .unwrap();

        let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
        let rendered = engine
            .apply_with_tools(
                r#"[{"role":"user","content":"Check London weather."},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"weather","arguments":"{\"city\":\"London\",\"days\":2}"}}]}]"#,
                r#"[{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]"#,
                false,
                false,
            )
            .unwrap();
        assert!(rendered.contains("<｜DSML｜invoke name=\"weather\">"));
        assert!(rendered.contains(
            "<｜DSML｜parameter name=\"city\" string=\"true\">London</｜DSML｜parameter>"
        ));
        assert!(rendered
            .contains("<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>"));
        assert!(rendered.contains("<｜end▁of▁sentence｜>"));
    }

    #[test]
    fn detects_each_shipped_native_tool_call_grammar_from_template_contract() {
        let cases = [
            (
                r#"<tool_call>{"name": <function-name>, "arguments": {}}</tool_call>"#,
                ToolCallFormat::QwenJson,
            ),
            (
                "<tool_call><function=example_function_name><parameter=x></parameter></function></tool_call>",
                ToolCallFormat::FunctionXml,
            ),
            (
                "<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>",
                ToolCallFormat::GlmXml,
            ),
            (
                "{% set dsml_token = '｜DSML｜' %}<invoke name=\"tool\">",
                ToolCallFormat::DeepseekDsml,
            ),
            (
                "<|tool_call>call:name{}<tool_call|>",
                ToolCallFormat::Gemma,
            ),
            (
                "{% set x = '<minimax:tool_call>' %}<invoke name=\"tool\">",
                ToolCallFormat::Minimax,
            ),
            ("User: {{ message.content }}", ToolCallFormat::Unsupported),
        ];
        for (template, expected) in cases {
            assert_eq!(detect_tool_call_format(template), expected, "{template}");
        }
    }
}
