use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "Usage: expert_hqq_row_attribution <response.json> <diagnostic_spec.json> <metrics.tsv> [details.json]"
    );
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(response_path) = args.next() else {
        usage();
    };
    let Some(spec_path) = args.next() else {
        usage();
    };
    let Some(metrics_path) = args.next() else {
        usage();
    };
    let details_path = args.next();
    if args.next().is_some() {
        usage();
    }

    let response_path = PathBuf::from(response_path);
    let spec_path = PathBuf::from(spec_path);
    let metrics_path = PathBuf::from(metrics_path);
    let details_path = details_path.map(PathBuf::from);

    match krasis::weights::expert_hqq::attribute_expert_hqq_exact_row_trace_paths(
        &response_path,
        &spec_path,
        &metrics_path,
        details_path.as_deref(),
    ) {
        Ok(report) => {
            println!("expert_hqq_row_attribution status=ok");
            println!("response={}", report.response_path.display());
            println!("spec={}", report.spec_path.display());
            println!("cache={}", report.cache_path.display());
            println!("layer={}", report.layer_idx);
            println!("requested_expert={}", report.requested_expert_idx);
            println!("requested_sorted_row={}", report.requested_sorted_row);
            println!("requested_col={}", report.requested_col);
            println!("captured_rows={}", report.captured_rows);
            println!("selected_contributors={}", report.selected_contributors);
            println!(
                "max_hqq_gpu_vs_krhq_output_abs={:.18e}",
                report.max_hqq_gpu_vs_krhq_output_abs
            );
            println!(
                "max_bf16_vs_krhq_output_abs={:.18e}",
                report.max_bf16_vs_krhq_output_abs
            );
            println!("selected_bf16_value={:.18e}", report.selected_bf16_value);
            println!(
                "selected_hqq_gpu_value={:.18e}",
                report.selected_hqq_gpu_value
            );
            println!("selected_krhq_value={:.18e}", report.selected_krhq_value);
            println!(
                "selected_bf16_vs_hqq_gpu_abs={:.18e}",
                report.selected_bf16_vs_hqq_gpu_abs
            );
            println!(
                "selected_bf16_vs_krhq_abs={:.18e}",
                report.selected_bf16_vs_krhq_abs
            );
            println!(
                "selected_hqq_gpu_vs_krhq_abs={:.18e}",
                report.selected_hqq_gpu_vs_krhq_abs
            );
            println!("attribution={}", report.attribution);
        }
        Err(err) => {
            eprintln!("expert_hqq_row_attribution status=error");
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
