use super::{api_keys::ApiKeys, parse_request, send_json};
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    server
        .set_read_timeout(Some(std::time::Duration::from_millis(250)))
        .unwrap();
    (server, client)
}

#[test]
fn http_headers_enforce_auth_before_body_and_handle_case_and_duplicates() {
    let path = std::env::temp_dir().join(format!("krasis-http-keys-{}.json", std::process::id()));
    std::fs::write(&path, r#"{"client":"test-secret"}"#).unwrap();
    let policy = ApiKeys::new(Some(path.to_str().unwrap())).unwrap();
    for headers in [
        "",
        "Authorization: Bearer wrong\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer test-secret\r\n",
    ] {
        let (server, mut client) = pair();
        // No body sent: auth must reject immediately instead of waiting for it.
        write!(
            client,
            "POST /v1/chat/completions HTTP/1.1\r\n{headers}Content-Length: 1000000\r\n\r\n"
        )
        .unwrap();
        let error = parse_request(&mut BufReader::new(server), &policy)
            .err()
            .unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    }
    let (server, mut client) = pair();
    write!(client, "POST /chat/completions HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR test-secret\r\ncOnTeNt-LeNgTh: 2\r\n\r\n{{}}").unwrap();
    let request = parse_request(&mut BufReader::new(server), &policy).unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/chat/completions");
    assert_eq!(request.body, "{}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unauthorized_response_is_standard_http_json() {
    let (mut server, mut client) = pair();
    send_json(&mut server, 401, r#"{"error":"Invalid API key"}"#).unwrap();
    drop(server);
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 401 Unauthorized\r\n"));
    assert!(response.contains("\r\nWWW-Authenticate: Bearer\r\n"));
}
