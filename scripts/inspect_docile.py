"""Inspect a DocILE ZIP remotely; download only selected archive members.

Set DOCILE_ACCESS_TOKEN in the process environment. Tokens and download URLs
are deliberately excluded from output and saved manifests.
"""

import argparse
import collections
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import zipfile


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP reader with bounded caching and verified byte ranges."""

    BLOCK = 1024 * 1024

    def __init__(self, token):
        self.url = (
            "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/"
            + token
            + "/annotated-trainval.zip"
        )
        with urllib.request.urlopen(
            urllib.request.Request(self.url, method="HEAD"), timeout=30
        ) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache = collections.OrderedDict()
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative archive position")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            block = self.position // self.BLOCK
            start = block * self.BLOCK
            if block not in self.cache:
                stop = min(self.size, start + self.BLOCK) - 1
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{stop}"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_range = f"bytes {start}-{stop}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Server did not honor the requested byte range")
                    data = response.read(stop - start + 2)
                if len(data) != stop - start + 1:
                    raise RuntimeError("Incomplete archive block")
                self.cache[block] = data
                self.transferred += len(data)
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            offset = self.position - start
            part = self.cache[block][offset : offset + end - self.position]
            chunks.append(part)
            self.position += len(part)
        return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--members", nargs="*")
    parser.add_argument("--output", type=Path, default=Path("data/docile_sample"))
    args = parser.parse_args()
    token = os.environ.get("DOCILE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Set DOCILE_ACCESS_TOKEN for this process")
    remote = RemoteZipFile(token)
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        if args.inventory:
            roots = collections.Counter(name.split("/")[0] for name in names)
            result = {
                "archive_bytes": remote.size,
                "entry_count": len(names),
                "root_entries": dict(roots),
                "first_names": names[:15],
                "split_manifests": {},
            }
            for name in names:
                if "/" not in name and name.endswith(".json"):
                    content = json.loads(archive.read(name))
                    result["split_manifests"][name] = {
                        "type": type(content).__name__,
                        "count": len(content),
                        "first": content[:5] if isinstance(content, list) else list(content)[:5],
                    }
            result["transferred_bytes"] = remote.transferred
            print(json.dumps(result, indent=2))
        if args.members:
            output = args.output.resolve()
            output.mkdir(parents=True, exist_ok=True)
            downloaded = []
            for name in args.members:
                target = (output / name).resolve()
                if not target.is_relative_to(output):
                    raise ValueError("Archive member is outside the output directory")
                content = archive.read(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                downloaded.append({"member": name, "bytes": len(content)})
            print(json.dumps({"downloaded": downloaded, "transferred_bytes": remote.transferred}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"DocILE request failed: HTTP {error.code}") from None
    except urllib.error.URLError:
        raise SystemExit("DocILE request failed: network error") from None
