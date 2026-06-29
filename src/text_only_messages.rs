pub(crate) fn validate_text_only_messages(messages: &serde_json::Value) -> Result<(), String> {
    let arr = messages
        .as_array()
        .ok_or_else(|| "messages must be an array".to_string())?;
    for (msg_idx, msg) in arr.iter().enumerate() {
        let Some(content) = msg.get("content") else {
            continue;
        };
        match content {
            serde_json::Value::String(_) | serde_json::Value::Null => {}
            serde_json::Value::Array(parts) => {
                for (part_idx, part) in parts.iter().enumerate() {
                    let obj = part.as_object().ok_or_else(|| {
                        format!(
                            "text-only runtime requires structured content part messages[{}].content[{}] to be an object",
                            msg_idx, part_idx
                        )
                    })?;
                    if is_multimodal_part(obj) {
                        return Err(format!(
                            "text-only runtime does not support image/video/audio/file content at messages[{}].content[{}]",
                            msg_idx, part_idx
                        ));
                    }
                    let part_type = obj.get("type").and_then(|v| v.as_str());
                    let text = obj.get("text");
                    if part_type == Some("text") || text.is_some() {
                        if text.map(|v| v.is_string()).unwrap_or(false) {
                            continue;
                        }
                        return Err(format!(
                            "text-only runtime requires messages[{}].content[{}].text to be a string",
                            msg_idx, part_idx
                        ));
                    }
                    return Err(format!(
                        "text-only runtime only supports string content or text content parts; unsupported part at messages[{}].content[{}]",
                        msg_idx, part_idx
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "text-only runtime only supports string content or text content parts at messages[{}].content",
                    msg_idx
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn messages_have_image_parts(messages: &serde_json::Value) -> bool {
    let Some(arr) = messages.as_array() else {
        return false;
    };
    arr.iter().any(|msg| {
        msg.get("content")
            .and_then(|content| content.as_array())
            .map(|parts| {
                parts
                    .iter()
                    .any(|part| part.as_object().map(is_image_part).unwrap_or(false))
            })
            .unwrap_or(false)
    })
}

pub(crate) fn validate_image_only_messages(messages: &serde_json::Value) -> Result<(), String> {
    let arr = messages
        .as_array()
        .ok_or_else(|| "messages must be an array".to_string())?;
    for (msg_idx, msg) in arr.iter().enumerate() {
        let Some(content) = msg.get("content") else {
            continue;
        };
        match content {
            serde_json::Value::String(_) | serde_json::Value::Null => {}
            serde_json::Value::Array(parts) => {
                for (part_idx, part) in parts.iter().enumerate() {
                    let obj = part.as_object().ok_or_else(|| {
                        format!(
                            "image runtime requires structured content part messages[{}].content[{}] to be an object",
                            msg_idx, part_idx
                        )
                    })?;
                    if is_video_audio_or_file_part(obj) {
                        return Err(format!(
                            "Qwen vision runtime supports images only; video/audio/file content is not supported at messages[{}].content[{}]",
                            msg_idx, part_idx
                        ));
                    }
                    if is_image_part(obj) {
                        continue;
                    }
                    let part_type = obj.get("type").and_then(|v| v.as_str());
                    let text = obj.get("text");
                    if part_type == Some("text") || text.is_some() {
                        if text.map(|v| v.is_string()).unwrap_or(false) {
                            continue;
                        }
                        return Err(format!(
                            "image runtime requires messages[{}].content[{}].text to be a string",
                            msg_idx, part_idx
                        ));
                    }
                    return Err(format!(
                        "image runtime only supports text and image content parts; unsupported part at messages[{}].content[{}]",
                        msg_idx, part_idx
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "image runtime only supports string content or text/image content parts at messages[{}].content",
                    msg_idx
                ));
            }
        }
    }
    Ok(())
}

fn is_image_part(part: &serde_json::Map<String, serde_json::Value>) -> bool {
    if part.contains_key("image") || part.contains_key("image_url") {
        return true;
    }
    matches!(
        part.get("type").and_then(|v| v.as_str()),
        Some("image" | "image_url" | "input_image")
    )
}

fn is_video_audio_or_file_part(part: &serde_json::Map<String, serde_json::Value>) -> bool {
    if part.contains_key("video") || part.contains_key("audio") || part.contains_key("file") {
        return true;
    }
    matches!(
        part.get("type").and_then(|v| v.as_str()),
        Some("video" | "input_video" | "audio" | "input_audio" | "file" | "input_file")
    )
}

fn is_multimodal_part(part: &serde_json::Map<String, serde_json::Value>) -> bool {
    if part.contains_key("image")
        || part.contains_key("image_url")
        || part.contains_key("video")
        || part.contains_key("audio")
        || part.contains_key("file")
    {
        return true;
    }
    matches!(
        part.get("type").and_then(|v| v.as_str()),
        Some(
            "image"
                | "image_url"
                | "input_image"
                | "video"
                | "input_video"
                | "audio"
                | "input_audio"
                | "file"
                | "input_file"
        )
    )
}

#[cfg(test)]
mod tests {
    use super::{validate_image_only_messages, validate_text_only_messages};

    #[test]
    fn accepts_strings_and_text_parts() {
        let messages = serde_json::json!([
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"text": " world"}
            ]},
            {"role": "assistant", "content": null}
        ]);
        validate_text_only_messages(&messages).unwrap();
    }

    #[test]
    fn rejects_multimodal_parts() {
        let messages = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]}
        ]);
        let err = validate_text_only_messages(&messages).unwrap_err();
        assert!(err.contains("does not support image/video/audio/file content"));
    }

    #[test]
    fn rejects_unknown_structured_parts() {
        let messages = serde_json::json!([
            {"role": "user", "content": [{"type": "tool_result", "result": "x"}]}
        ]);
        let err = validate_text_only_messages(&messages).unwrap_err();
        assert!(err.contains("only supports string content or text content parts"));
    }

    #[test]
    fn accepts_image_parts_for_image_only_runtime() {
        let messages = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]}
        ]);
        validate_image_only_messages(&messages).unwrap();
    }

    #[test]
    fn rejects_video_parts_for_image_only_runtime() {
        let messages = serde_json::json!([
            {"role": "user", "content": [
                {"type": "input_video", "video": "file:///tmp/a.mp4"}
            ]}
        ]);
        let err = validate_image_only_messages(&messages).unwrap_err();
        assert!(err.contains("supports images only"));
    }
}
