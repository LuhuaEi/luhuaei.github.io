    WARNING: The vboxdrv kernel module is not loaded. Either there is no module
             available for the current kernel (xxx) or it failed to
             load. Please recompile the kernel module and install it by

               sudo /etc/init.d/vboxdrv setup

             You will not be able to start VMs until this problem is fixed.

刚开始还以为是Arch的内核太新了，导致virtualbox不支持，后面将过切换到低级的内核还
是不能解决这个问题。最后在[stackover
flow](https://stackoverflow.com/questions/23740932/warning-vboxdrv-kernel-module-is-not-loaded)上面找到了一个解决方案。

    sudo pacman -S virtualbox-host-modules-arch
    sudo modprobe vboxdrv

这样就可以解决上面出现的错误了。

    Stderr: 0%...
    Progress state: NS_ERROR_FAILURE
    VBoxManage: error: Failed to create the host-only adapter
    VBoxManage: error: VBoxNetAdpCtl: Error while adding new interface: VBoxNetAdpCtl: ioctl failed for /dev/vboxnetctl: Inappropriate ioctl for devic
    VBoxManage: error: Details: code NS_ERROR_FAILURE (0x80004005), component HostNetworkInterface, interface IHostNetworkInterface
    VBoxManage: error: Context: "int handleCreate(HandlerArg*, int, int*)" at line
    66 of file VBoxManageHostonly.cpp

这个问题是host地址的引起的错误，在linux系统下面执行
`sudo vboxreload`
