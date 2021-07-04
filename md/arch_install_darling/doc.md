# 编译darling

```sh
Cannot load libavresample.so: libavresample.so: cannot open shared object file: No such file or directory
make[2]: *** [src/CoreAudio/CMakeFiles/avresample.dir/build.make:74: src/CoreAudio/avresample.c] Error 1
make[2]: *** Deleting file 'src/CoreAudio/avresample.c'
make[1]: *** [CMakeFiles/Makefile2:21784: src/CoreAudio/CMakeFiles/avresample.dir/all] Error 2
make: *** [Makefile:136: all] Error 2
```

可以通过安装 `libavresample` 解决这个问题, 本质是在编译`darling`的时候找不到该动态库，然而最新版的`ffmpeg`已经将`--enable-avresample`这个选项去掉了，而`Arch AUR`上的拉取的版本还比较旧。

```sh
yaourt libavresample
```

# 编译darling lkm

```sh
not found /lib/modules/5.12.14-arch1-1/build directory
```
一个是需要安装 `linux-header`, 安装完后，可能会要重启一下，因为`Arch`是滚动更新，`uname -r` 的版本跟目前安装的版本不一样.

# attach DMG 文件失败

不知道什么原因，但`draling shell`中执行 `hdiutil attach xxx`会提示一下错误

```sh
Skipping partition of type Primary GPT Header
Skipping partition of type Primary GPT Table
Skipping partition of type Apple_Free
Using partition #3 of type Apple_HFS
Error: Unexpected EOF from readRun

Possible reasons:
1) The file is corrupt.
2) The file is not really a DMG file, although it resembles one.
3) There is a bug in darling-dmg.
```

这个可以将 `DMG`文件先使用 `7z x xxxx.dmg` 解压出来，然后将`xxx.app`复制到对应的`/Application/`目录下即可。

但发现darling下面的`Wechat`会报

```sh
dyld: Library not loaded: /System/Library/Frameworks/CoreTelephony.framework/Versions/A/CoreTelephony
  Referenced from: /Applications/WeChat.app/Contents/Frameworks/ilink_network.framework/Versions/A/ilink_network
  Reason: image not found
abort_with_payload: reason: Library not loaded: /System/Library/Frameworks/CoreTelephony.framework/Versions/A/CoreTelephony
  Referenced from: /Applications/WeChat.app/Contents/Frameworks/ilink_network.framework/Versions/A/ilink_network
  Reason: image not found; code: 1
Abort trap: 6 (core dumped)
```

在源码里找了下， 目前是不支持`CoreTelephony.framework`的


QQ 会出现以下错误

```sh
dyld: Symbol not found: _kCIInputAspectRatioKey
  Referenced from: /Applications/QQ.app/Contents/MacOS/../Frameworks/MacQQUI.framework/Versions/A/MacQQUI
  Expected in: /System/Library/Frameworks/QuartzCore.framework/Versions/A/QuartzCore
 in /Applications/QQ.app/Contents/MacOS/../Frameworks/MacQQUI.framework/Versions/A/MacQQUI
abort_with_payload: reason: Symbol not found: _kCIInputAspectRatioKey
  Referenced from: /Applications/QQ.app/Contents/MacOS/../Frameworks/MacQQUI.framework/Versions/A/MacQQUI
  Expected in: /System/Library/Frameworks/QuartzCore.framework/Versions/A/QuartzCore
 in /Applications/QQ.app/Contents/MacOS/../Frameworks/MacQQUI.framework/Versions/A/MacQQUI; code: 4
Abort trap: 6 (core dumped)
```

还是不能用
