# QEMU (Virtual machine)

You have the option of QEMU/KVM instead of the usual VirtualBox.

"QEMU is a generic and open source machine emulator and virtualizer." [...] "QEMU can use other hypervisors like Xen or KVM to use CPU extensions (HVM) for virtualization. When used as a virtualizer, QEMU achieves near native performances by executing the guest code directly on the host CPU."

"Unlike native QEMU, which uses emulation, KVM is a special operating mode of QEMU that uses CPU extensions (HVM) for virtualization via a kernel module."

Do the following:

```sh
sudo pacman -S qemu
qemu-img create -f qcow2 image.img 16G
qemu-system-x86_64 -enable-kvm -cdrom ubuntu.iso -boot d -drive file=image.img
-m 2G
```

Notes:

* `-f qcow2`: format of virtual disk, supporting sparse file systems, AES encryption, zlib-based compression and multiple VM snapshots (CoW=Copy on Write, q=qemu, 2=version2)
* `-m 2G` 2048MB of RAM
* `-boot d` boot from cdrom (iso), remove this if image has already been
  installed on `image.img`
* `-enable-kvm` enables KVM (see above), make sure you have the loaded kernel
  modules (`lsmod | grep kvm`)

- `-no-acpi` (disable the disfuncional ACPI) was cutting internet connection on
  debian VM

Additional flags:

- `-boot menu=on` give a menu to select bootable device
- `-monitor stdio` do not use graphics
- `-cpu host` use the host cpu, instead of an emulated one
- `-smp 4` set 4 cores
- `-vga qxl bochs_drm` accelerate graphics (use `sudo modprobe qxl bochs_drm`)
- `-vga virtio` even better graphics (mandatory to the `-display sdl,gl=on`)
- `-display sdl,gl=on` enable OpenGL

In graphical mode:

- to escape with the mouse: ctrl+alt+g
- to make it full screen: ctrl+alt+f

If you are in a Linux system, with a Windows machine inside:

* Spice Guest Tools are used for:
  * speed-up the VM
  * to share folders, you have to install (install daemon inside Windows)

## resizing qcow2 image

```sh
qemu-image resize my_image.qcow2 50G
```

Output should be `image resized`.

## check if KVM is installed

Alternative to grepping lsmod, you can use `cpu-checker`:

```sh
paru -S cpu-checker
kvm-ok
```

## Networking

- [QEMU wiki - Networking](https://wiki.qemu.org/Documentation/Networking)
