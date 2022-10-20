# Mitigating Domain Shift in AI-Based TB Screening
With Unsupervised Domain Adaptation
Paper Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9759448

Requirements:
Pytorch
PIL

Dataset Directory
.
├── ...
├── domain 1      # Domain Folder (China, India etc)
│   ├── 0         # Negative Examples
│   ├── 1         # Positive Examples
├── domain 2      # Domain Folder (China, India etc)
│   ├── 0         # Negative Examples
│   ├── 1         # Positive Examples
└── ...


$ ./tree-md .
# Dataset Directory

.
 * [tree-md](./tree-md)
 * [dir2](./dir2)
   * [file21.ext](./dir2/file21.ext)
   * [file22.ext](./dir2/file22.ext)
   * [file23.ext](./dir2/file23.ext)
 * [dir1](./dir1)
   * [file11.ext](./dir1/file11.ext)
   * [file12.ext](./dir1/file12.ext)
 * [file_in_root.ext](./file_in_root.ext)
 * [README.md](./README.md)
 * [dir3](./dir3)
