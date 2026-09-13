import tomllib
import unittest
from pathlib import Path


class DeploymentDependenciesTest(unittest.TestCase):
    def test_arm_pytorch_wheels_do_not_require_cuda(self):
        lock = tomllib.loads(Path("uv.lock").read_text())
        cpu_packages = [
            package for package in lock["package"]
            if package["name"] in ("torch", "torchvision")
            and package.get("source", {}).get("registry") == "https://download.pytorch.org/whl/cpu"
        ]
        self.assertEqual({p["name"] for p in cpu_packages}, {"torch", "torchvision"})
        for package in cpu_packages:
            self.assertTrue(any("aarch64" in wheel["url"] for wheel in package["wheels"]))
            for dependency in package.get("dependencies", []):
                self.assertFalse(dependency["name"].startswith(("cuda-", "nvidia-", "triton")))


if __name__ == "__main__":
    unittest.main()
