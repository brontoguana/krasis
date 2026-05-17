from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_stdout_tee_uses_file_only_logging():
    source = (ROOT / "python/krasis/server.py").read_text()
    setup = source[source.index("log_format = "):source.index('logger.info("Logging to %s"')]

    assert "logging.basicConfig" not in setup
    assert "StreamHandler" not in setup
    assert "_root_logger.addHandler(_file_handler)" in setup
    assert "sys.stdout = _StreamLogger(sys.stdout, logger.info)" in setup
    assert "sys.stderr = _StreamLogger(sys.stderr, logger.error)" in setup


def test_vram_monitor_operator_warnings_are_single_terminal_path():
    source = (ROOT / "src/vram_monitor.rs").read_text()
    hard_exit = source[
        source.index('append_safety_limit_dump(\n                                        "hard_exit_floor"'):
        source.index("// The hard floor means CUDA is already in an unsafe")
    ]
    below_safety = source[
        source.index('append_safety_limit_dump(\n                                            "below_safety_margin"'):
        source.index('eprintln!(\n                                            "\\x1b[1;33m')
    ]

    assert "log::error!" not in hard_exit
    assert "log::warn!" not in below_safety
