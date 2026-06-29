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
fn text_content_parts_are_flattened_for_templates() {
    let template = "{% for message in messages %}{{ message.content }}{% endfor %}";
    let config_path = write_tokenizer_config(template, "text_parts");
    let engine = ChatTemplateEngine::from_config(&config_path).unwrap();
    let messages =
        r#"[{"role":"user","content":[{"type":"text","text":"Hello"},{"text":" world"}]}]"#;
    assert_eq!(engine.apply(messages, false, false).unwrap(), "Hello world");
}
