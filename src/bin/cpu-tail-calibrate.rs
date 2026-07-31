fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = krasis::cpu_tail_calibrate::run_cli(&args) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
