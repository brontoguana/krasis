#[path = "../src/chat_template.rs"]
mod chat_template;

use chat_template::ChatTemplateEngine;
use std::fs;

fn write_tokenizer_config(template: &str, name: &str) -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "krasis_chat_template_integration_{}_{}_{}",
        std::process::id(),
        name,
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

fn write_architecture_only_tokenizer_config(model_type: &str, name: &str) -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "krasis_chat_template_arch_test_{}_{}_{}",
        std::process::id(),
        name,
        nonce
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        serde_json::json!({"model_type": model_type}).to_string(),
    )
    .unwrap();
    let path = dir.join("tokenizer_config.json");
    fs::write(
        &path,
        serde_json::json!({
            "chat_template": null,
            "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
            "eos_token": {"content": "<｜end▁of▁sentence｜>"}
        })
        .to_string(),
    )
    .unwrap();
    path.to_string_lossy().to_string()
}

#[test]
fn qwen36_preserve_thinking_is_defined_false() {
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
    let config_path = write_tokenizer_config(template, "preserve");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let rendered = engine
        .apply(
            r#"[{"role":"assistant","content":"old reasoning"}]"#,
            false,
            false,
        )
        .unwrap();
    assert_eq!(rendered, "DROP:old reasoning");
}

#[test]
fn qwen36_thinking_generation_prompt_modes() {
    let template = concat!(
        "{% if add_generation_prompt %}",
        "{% if enable_thinking is defined and enable_thinking is false %}",
        "<think>\n\n</think>\n\n",
        "{% else %}",
        "<think>\n",
        "{% endif %}",
        "{% endif %}"
    );
    let config_path = write_tokenizer_config(template, "thinking");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let messages = r#"[{"role":"user","content":"hi"}]"#;
    assert_eq!(
        engine.apply(messages, true, false).unwrap(),
        "<think>\n\n</think>\n\n"
    );
    assert_eq!(engine.apply(messages, true, true).unwrap(), "<think>\n");
}

#[test]
fn unconditional_terminal_think_marker_closes_when_disabled() {
    let template = concat!(
        "{% for message in messages %}",
        "<|{{ message.role }}|>{{ message.content }}",
        "{% endfor %}",
        "{% if add_generation_prompt %}<|assistant|><think>{% endif %}"
    );
    let config_path = write_tokenizer_config(template, "unconditional-thinking");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let messages = r#"[{"role":"user","content":"hi"}]"#;

    assert_eq!(
        engine.apply(messages, true, false).unwrap(),
        "<|user|>hi<|assistant|><think></think>"
    );
    assert_eq!(
        engine.apply(messages, true, true).unwrap(),
        "<|user|>hi<|assistant|><think>"
    );
}

#[test]
fn text_content_parts_are_flattened_for_templates() {
    let template = "{% for message in messages %}{{ message.content }}{% endfor %}";
    let config_path = write_tokenizer_config(template, "text_parts");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let messages =
        r#"[{"role":"user","content":[{"type":"text","text":"Hello"},{"text":" world"}]}]"#;
    assert_eq!(engine.apply(messages, false, false).unwrap(), "Hello world");
}

#[test]
fn deepseek_v4_without_checkpoint_jinja_uses_architecture_template() {
    let config_path = write_architecture_only_tokenizer_config("deepseek_v4", "dsv4");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let rendered = engine
        .apply(
            r#"[{"role":"user","content":"Give the chemical symbol for sodium, followed by one short sentence."}]"#,
            true,
            false,
        )
        .unwrap();
    assert_eq!(
        rendered,
        "<｜begin▁of▁sentence｜><｜User｜>Give the chemical symbol for sodium, followed by one short sentence.<｜Assistant｜></think>"
    );
}

#[test]
fn deepseek_v4_renders_openai_tool_calls() {
    let config_path = write_architecture_only_tokenizer_config("deepseek_v4", "dsv4_tools");
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
    assert!(rendered
        .contains("<｜DSML｜parameter name=\"city\" string=\"true\">London</｜DSML｜parameter>"));
    assert!(rendered
        .contains("<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>"));
    assert!(rendered.contains("<｜end▁of▁sentence｜>"));
}

#[test]
fn from_json_filter_rejects_malformed_tool_arguments() {
    let template = "{{ messages[0].content | from_json }}";
    let config_path = write_tokenizer_config(template, "from_json_invalid");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let error = engine
        .apply(r#"[{"role":"user","content":"{broken"}]"#, false, false)
        .unwrap_err();
    assert!(error.contains("from_json received invalid JSON"));
}
