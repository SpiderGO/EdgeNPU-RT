#ifndef TINYNPU_IOCTL_H
#define TINYNPU_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define TINYNPU_IOC_MAGIC 'N'

struct tinynpu_gemm_submit {
    __u32 a_handle;
    __u32 b_handle;
    __u32 c_handle;

    __u32 m;
    __u32 n;
    __u32 k;

    __u32 dtype;
    __u32 flags;

    __u64 sequence;
};

struct tinynpu_wait_request {
    __u64 sequence;
    __u32 status;
    __u32 reserved;
};

struct tinynpu_buffer_register {
    __u64 user_addr;
    __u64 size;
    __u32 flags;
    __u32 handle;
};

struct tinynpu_buffer_release {
    __u32 handle;
    __u32 reserved;
};

#define TINYNPU_IOCTL_SUBMIT_GEMM \
    _IOWR(TINYNPU_IOC_MAGIC, 0x01, struct tinynpu_gemm_submit)

#define TINYNPU_IOCTL_WAIT \
    _IOWR(TINYNPU_IOC_MAGIC, 0x02, struct tinynpu_wait_request)

#define TINYNPU_IOCTL_REGISTER_BUFFER \
    _IOWR(TINYNPU_IOC_MAGIC, 0x03, struct tinynpu_buffer_register)

#define TINYNPU_IOCTL_RELEASE_BUFFER \
    _IOW(TINYNPU_IOC_MAGIC, 0x04, struct tinynpu_buffer_release)
    
#endif