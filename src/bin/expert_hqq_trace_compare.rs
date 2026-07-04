use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "Usage: expert_hqq_trace_compare <trace_outputs.json> <diagnostic_spec.json> <metrics.tsv> [mismatch_details.tsv [failure_rows.tsv]]"
    );
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(trace_path) = args.next() else {
        usage();
    };
    let Some(spec_path) = args.next() else {
        usage();
    };
    let Some(metrics_path) = args.next() else {
        usage();
    };
    let mismatch_details_path = args.next();
    let failure_rows_path = args.next();
    if args.next().is_some() {
        usage();
    }
    if failure_rows_path.is_some() && mismatch_details_path.is_none() {
        usage();
    }

    let trace_path = PathBuf::from(trace_path);
    let spec_path = PathBuf::from(spec_path);
    let metrics_path = PathBuf::from(metrics_path);
    let mismatch_details_path = mismatch_details_path.map(PathBuf::from);
    let failure_rows_path = failure_rows_path.map(PathBuf::from);

    let result = if let Some(failure_rows_path) = failure_rows_path.as_deref() {
        krasis::weights::expert_hqq::compare_expert_hqq_runtime_prefill_trace_paths_filtered_by_failure_rows(
            &trace_path,
            &spec_path,
            failure_rows_path,
            Some(&metrics_path),
            mismatch_details_path.as_deref(),
        )
    } else {
        krasis::weights::expert_hqq::compare_expert_hqq_runtime_prefill_trace_paths_with_mismatch_details(
            &trace_path,
            &spec_path,
            Some(&metrics_path),
            mismatch_details_path.as_deref(),
        )
    };

    match result {
        Ok(report) => {
            let (input, w13, activation, output) = report.stage_totals();
            println!("expert_hqq_trace_compare status=ok");
            println!("trace={}", report.trace_path.display());
            println!("spec={}", report.spec_path.display());
            println!("cache={}", report.cache_path.display());
            println!("cases={}", report.case_count());
            println!("blocks={}", report.block_count());
            println!("passes_contract={}", report.passes_contract);
            println!("input_sum_abs={:.18e}", input.sum_abs);
            println!("input_max_abs={:.18e}", input.max_abs);
            println!("w13_sum_abs={:.18e}", w13.sum_abs);
            println!("w13_max_abs={:.18e}", w13.max_abs);
            println!("activation_sum_abs={:.18e}", activation.sum_abs);
            println!("activation_max_abs={:.18e}", activation.max_abs);
            println!("output_sum_abs={:.18e}", output.sum_abs);
            println!("output_max_abs={:.18e}", output.max_abs);
            if !report.passes_contract {
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("expert_hqq_trace_compare status=error");
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
