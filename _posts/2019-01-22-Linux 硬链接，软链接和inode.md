---
layout:     post
title:      Linux 硬链接、软链接和iNode
subtitle:   APUE学习笔记
date:       2019-01-22
author:     Xiya Lv
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - APUE
    - iNode
    - 链接
---

## 硬链接与软链接

文件都有文件名与数据，这在 Linux 上被分成两个部分：用户数据 (user data) 与元数据 (metadata)。用户数据，即文件数据块 (data block)，数据块是记录文件真实内容的地方；而元数据则是文件的附加属性，如文件大小、创建时间、所有者等信息。

在 Linux 中，元数据中的 inode 号（inode 是文件元数据的一部分但其并不包含文件名，inode 号即索引节点号）才是文件的唯一标识而非文件名。文件名仅是为了方便人们的记忆和使用，系统或程序通过 inode 号寻找正确的文件数据块。[图 1.](https://www.ibm.com/developerworks/cn/linux/l-cn-hardandsymb-links/index.html#fig1)展示了程序通过文件名获取文件内容的过程。

##### 图 1. 通过文件名打开文件

![å¾ 1. éè¿æä"¶åæå¼æä"¶](https://www.ibm.com/developerworks/cn/linux/l-cn-hardandsymb-links/image001.jpg)

在 Linux 系统中查看 inode 号可使用命令` stat `或 `ls -i`.

为解决文件的共享使用，Linux 系统引入了两种链接：硬链接 (hard link) 与软链接（又称符号链接，即 soft link 或 symbolic link）。链接为 Linux 系统解决了文件的共享使用，还带来了隐藏文件路径、增加权限安全及节省存储等好处。



**硬链接**：若一个 inode 号对应多个文件名，则称这些文件为硬链接。换言之，**硬链接就是同一个文件使用了多个别名**（见 [图 2.](https://www.ibm.com/developerworks/cn/linux/l-cn-hardandsymb-links/index.html#fig2)hard link 就是 file 的一个别名，他们有共同的 inode）。硬链接可由命令 link 或 ln 创建。如下是对文件 oldfile 创建硬链接。

```
`link oldfile newfile ``ln oldfile newfile`
```

由于硬链接是有着相同 inode 号仅文件名不同的文件，因此硬链接存在以下几点特性：

- 文件有相同的 inode 及 data block；
- 只能对已存在的文件进行创建；
- 不能交叉文件系统进行硬链接的创建；
- 不能对目录进行创建，只可对文件创建；
- 删除一个硬链接文件并不影响其他有相同 inode 号的文件。

- 对一个iNode创建一个硬链接会增加其链接计数，当链接计数为0时，表示该文件可被删除。

Linux 系统存在 inode 号被用完但磁盘空间还有剩余的情况。

硬链接不能对目录创建是受限于文件系统的设计。现 Linux 文件系统中的目录均隐藏了两个个特殊的目录：当前目录（.）与父目录（..）。查看这两个特殊目录的 inode 号可知其实这两目录就是两个硬链接。若系统允许对目录创建硬链接，则会产生目录环。

**软链接**：若文件用户数据块中存放的内容是另一文件的路径名的指向，则该文件就是软连接。软链接就是一个普通文件，只是数据块内容有点特殊。软链接有着自己的 inode 号以及用户数据块（见 [图 2.](https://www.ibm.com/developerworks/cn/linux/l-cn-hardandsymb-links/index.html#fig2)）。因此软链接的创建与使用没有类似硬链接的诸多限制：

- 软链接有自己的文件属性及权限等；
- 可对不存在的文件或目录创建软链接；
- 软链接可交叉文件系统；
- 软链接可对文件或目录创建；
- 创建软链接时，链接计数 i_nlink 不会增加；
- 删除软链接并不影响被指向的文件，但若被指向的原文件被删除，则相关软连接被称为死链接（即 dangling link，若被指向路径文件被重新创建，死链接可恢复为正常的软链接）。

##### 图 2. 软链接的访问

![å¾ 2. è½¯é¾æ¥çè®¿é®](https://www.ibm.com/developerworks/cn/linux/l-cn-hardandsymb-links/image002.jpg)

## iNode

每个文件存在两个计数器：i_count 与 i_nlink，即**引用计数与硬链接计数**。

结构体 inode 中的 i_count 用于跟踪文件被访问的数量，而 i_nlink 则是上述使用 ls -l 等命令查看到的文件硬链接数。或者说 i_count 跟踪文件在内存中的情况，而 i_nlink 则是磁盘计数器。当文件被删除时，则 i_nlink 先被设置成 0。文件的这两个计数器使得 Linux 系统升级或程序更新变的容易。系统或程序可在不关闭的情况下（即文件 i_count 不为 0），将新文件以同样的文件名进行替换，新文件有自己的 inode 及 data block，旧文件会在相关进程关闭后被完整的删除。

### 参考文献：

1. 摘自 https://www.ibm.com/developerworks/cn/linux/l-cn-hardandsymb-links/index.html

















