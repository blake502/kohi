#pragma once

#include "defines.h"
#include "assets/kasset_types.h"

struct kpackage;
struct kasset_name;

typedef struct vfs_state {
    // darray
    struct kpackage* packages;
} vfs_state;

typedef struct vfs_config {
    const char** text_user_types;
} vfs_config;

typedef enum vfs_asset_flag_bits {
    VFS_ASSET_FLAG_NONE = 0,
    VFS_ASSET_FLAG_BINARY_BIT = 0x01
} vfs_asset_flag_bits;

typedef u32 vfs_asset_flags;

typedef struct vfs_asset_data {
    u64 size;
    union {
        const char* text;
        void* bytes;
    };
    vfs_asset_flags flags;

    b8 success;
} vfs_asset_data;

typedef void (*PFN_on_asset_loaded_callback)(const char* name, vfs_asset_data asset_data);

KAPI b8 vfs_initialize(u64* memory_requirement, vfs_state* out_state, const vfs_config* config);
KAPI void vfs_shutdown(vfs_state* state);

KAPI void vfs_request_asset(vfs_state* state, const struct kasset_name* name, kasset_type type, PFN_on_asset_loaded_callback callback);
