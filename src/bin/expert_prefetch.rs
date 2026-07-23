fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = krasis::expert_prefetch::run_cli(&args) {
        eprintln!("{}", e);
        std::process::exit(2);
    }
}
