import unittest

from krasis.ssh_tunnel import build_ssh_tunnel_command, parse_ssh_tunnel_target


class SshTunnelTests(unittest.TestCase):
    def test_parse_target_accepts_user_host_and_ssh_port(self):
        plain = parse_ssh_tunnel_target("alice@example.com")
        self.assertEqual(plain.destination, "alice@example.com")
        self.assertIsNone(plain.ssh_port)

        with_port = parse_ssh_tunnel_target("alice@example.com:2222")
        self.assertEqual(with_port.destination, "alice@example.com")
        self.assertEqual(with_port.ssh_port, 2222)

    def test_parse_target_rejects_shell_like_values(self):
        for target in ("", "example.com", "alice@", "-oProxyCommand=bad", "alice@example.com -v", "alice@example.com:notaport"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    parse_ssh_tunnel_target(target)

    def test_build_command_uses_loopback_reverse_forward_and_batch_mode(self):
        cmd = build_ssh_tunnel_command(
            "alice@example.com:2222",
            local_port=8012,
            key_path="~/.ssh/id_ed25519",
        )
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("BatchMode=yes", cmd)
        self.assertIn("IdentitiesOnly=yes", cmd)
        self.assertIn("ExitOnForwardFailure=yes", cmd)
        self.assertIn("ServerAliveInterval=30", cmd)
        self.assertIn("ServerAliveCountMax=3", cmd)
        self.assertIn("127.0.0.1:8012:127.0.0.1:8012", cmd)
        self.assertIn("-i", cmd)
        self.assertIn("/.ssh/id_ed25519", cmd[cmd.index("-i") + 1])
        self.assertIn("-p", cmd)
        self.assertIn("2222", cmd)
        self.assertEqual(cmd[-1], "alice@example.com")


if __name__ == "__main__":
    unittest.main()
