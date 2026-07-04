use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(manifest) = args.next() else {
        eprintln!("Usage: expert_hqq_cache_generate <manifest.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("Usage: expert_hqq_cache_generate <manifest.json>");
        std::process::exit(2);
    }

    match krasis::weights::expert_hqq::generate_expert_hqq_cache_from_manifest_path(&PathBuf::from(
        manifest,
    )) {
        Ok(report) => {
            println!("expert_hqq_cache_generate status=ok");
            println!("manifest={}", report.manifest_path.display());
            println!("cache={}", report.cache_path.display());
            println!("diagnostic_spec={}", report.diagnostic_spec_path.display());
            println!("layer_idx={}", report.layer_idx);
            println!("layer_count={}", report.layers.len());
            println!(
                "layers={}",
                report
                    .layers
                    .iter()
                    .map(|layer| layer.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            println!("expert_count={}", report.expert_count);
            println!("tensor_records={}", report.tensor_records);
            println!("total_payload_bytes={}", report.total_payload_bytes);
            println!("cache_file_bytes={}", report.cache_file_bytes);
        }
        Err(err) => {
            eprintln!("expert_hqq_cache_generate status=error");
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
