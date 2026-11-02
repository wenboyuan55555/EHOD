# 数据下载 / Dataset downloads

## 点击下面两个链接下载完整数据

- **[下载纽约数据：ehod-data-v1-nyc.zip（约 935 MB）](https://github.com/wenboyuan55555/EHOD/releases/download/data-v1/ehod-data-v1-nyc.zip)**
- **[下载芝加哥数据：ehod-data-v1-chicago.zip（约 22 MB）](https://github.com/wenboyuan55555/EHOD/releases/download/data-v1/ehod-data-v1-chicago.zip)**

两个 ZIP 共包含 28 个原始数据文件，解压后约 13.36 GB。
请将它们解压到 **EHOD 项目根目录**，文件会自动放入 `data/nyc/` 和
`data/chicago/`。

**[打开 GitHub Release 发布页，查看全部附件](https://github.com/wenboyuan55555/EHOD/releases/tag/data-v1)**

发布页底部的 **Assets** 列出了两个 ZIP、`SHA256SUMS` 校验清单以及数据包说明。
本文件夹中的 [data-v1.json](data-v1.json) 记录下载地址、大小和 SHA-256 校验值。

## 自动下载并校验

在项目根目录运行：

```bash
python scripts/download_data.py --city all
```

Download both ZIP archives above and extract them into the repository root.
Alternatively, run the command above to download, extract, and verify all data
files automatically. See [the dataset guide](../README.md) for file schemas,
provenance, and verification instructions.
