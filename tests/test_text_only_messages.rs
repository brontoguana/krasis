#[path = "../src/text_only_messages.rs"]
mod text_only_messages;

use text_only_messages::validate_text_only_messages;

#[test]
fn accepts_openai_text_content_parts() {
    let messages = serde_json::json!([
        {"role": "system", "content": "Use short answers."},
        {"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"text": " there"}
        ]}
    ]);
    validate_text_only_messages(&messages).unwrap();
}

#[test]
fn rejects_openai_image_content_parts() {
    let messages = serde_json::json!([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]}
    ]);
    let err = validate_text_only_messages(&messages).unwrap_err();
    assert!(err.contains("does not support image/video/audio/file content"));
}

#[test]
fn rejects_structured_non_text_parts() {
    let messages = serde_json::json!([
        {"role": "user", "content": [{"type": "input_audio", "audio": "abc"}]}
    ]);
    let err = validate_text_only_messages(&messages).unwrap_err();
    assert!(err.contains("does not support image/video/audio/file content"));
}
