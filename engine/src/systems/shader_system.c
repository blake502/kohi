#include "shader_system.h"

#include "containers/darray.h"
#include "core/frame_data.h"
#include "core/kmemory.h"
#include "core/kstring.h"
#include "core/logger.h"
#include "defines.h"
#include "renderer/renderer_frontend.h"
#include "renderer/renderer_utils.h"
#include "resources/resource_types.h"
#include "systems/texture_system.h"

// The internal shader system state.
typedef struct shader_system_state {
    // This system's configuration.
    shader_system_config config;
    // A lookup table for shader name->id
    hashtable lookup;
    // The memory used for the lookup table.
    void* lookup_memory;
    // The identifier for the currently bound shader.
    u32 current_shader_id;
    // A collection of created shaders.
    shader* shaders;
} shader_system_state;

// A pointer to hold the internal system state.
static shader_system_state* state_ptr = 0;

static b8 internal_attribute_add(shader* shader, const shader_attribute_config* config);
static b8 internal_sampler_add(shader* shader, shader_uniform_config* config);
static u32 generate_new_shader_id(void);
static b8 internal_uniform_add(shader* shader, const shader_uniform_config* config, u32 location);
static b8 uniform_name_valid(shader* shader, const char* uniform_name);
static b8 shader_uniform_add_state_valid(shader* shader);
static void internal_shader_destroy(shader* s);
static b8 sampler_state_try_set(shader_uniform_sampler_state* sampler_uniforms, u32 sampler_count, u16 uniform_location, u32 array_index, texture_map* map);

///////////////////////

b8 shader_system_initialize(u64* memory_requirement, void* memory, void* config) {
    shader_system_config* typed_config = (shader_system_config*)config;
    // Verify configuration.
    if (typed_config->max_shader_count < 512) {
        if (typed_config->max_shader_count == 0) {
            KERROR("shader_system_initialize - config.max_shader_count must be greater than 0");
            return false;
        } else {
            // This is to help avoid hashtable collisions.
            KWARN("shader_system_initialize - config.max_shader_count is recommended to be at least 512.");
        }
    }

    // Figure out how large of a hashtable is needed.
    // Block of memory will contain state structure then the block for the hashtable.
    u64 struct_requirement = sizeof(shader_system_state);
    u64 hashtable_requirement = sizeof(u32) * typed_config->max_shader_count;
    u64 shader_array_requirement = sizeof(shader) * typed_config->max_shader_count;
    *memory_requirement = struct_requirement + hashtable_requirement + shader_array_requirement;

    if (!memory) {
        return true;
    }

    // Setup the state pointer, memory block, shader array, then create the hashtable.
    state_ptr = memory;
    u64 addr = (u64)memory;
    state_ptr->lookup_memory = (void*)(addr + struct_requirement);
    state_ptr->shaders = (void*)((u64)state_ptr->lookup_memory + hashtable_requirement);
    state_ptr->config = *typed_config;
    state_ptr->current_shader_id = INVALID_ID;
    hashtable_create(sizeof(u32), typed_config->max_shader_count, state_ptr->lookup_memory, false, &state_ptr->lookup);

    // Invalidate all shader ids.
    for (u32 i = 0; i < typed_config->max_shader_count; ++i) {
        state_ptr->shaders[i].id = INVALID_ID;
        state_ptr->shaders[i].render_frame_number = INVALID_ID_U64;
    }

    // Fill the table with invalid ids.
    u32 invalid_fill_id = INVALID_ID;
    if (!hashtable_fill(&state_ptr->lookup, &invalid_fill_id)) {
        KERROR("hashtable_fill failed.");
        return false;
    }

    for (u32 i = 0; i < state_ptr->config.max_shader_count; ++i) {
        state_ptr->shaders[i].id = INVALID_ID;
    }

    return true;
}

void shader_system_shutdown(void* state) {
    if (state) {
        // Destroy any shaders still in existence.
        shader_system_state* st = (shader_system_state*)state;
        for (u32 i = 0; i < st->config.max_shader_count; ++i) {
            shader* s = &st->shaders[i];
            if (s->id != INVALID_ID) {
                internal_shader_destroy(s);
            }
        }
        hashtable_destroy(&st->lookup);
        kzero_memory(st, sizeof(shader_system_state));
    }

    state_ptr = 0;
}

b8 shader_system_create(renderpass* pass, const shader_config* config) {
    u32 id = generate_new_shader_id();
    shader* out_shader = &state_ptr->shaders[id];
    kzero_memory(out_shader, sizeof(shader));
    out_shader->id = id;
    if (out_shader->id == INVALID_ID) {
        KERROR("Unable to find free slot to create new shader. Aborting.");
        return false;
    }
    out_shader->state = SHADER_STATE_NOT_CREATED;
    out_shader->name = string_duplicate(config->name);
    out_shader->local_ubo_offset = 0;
    out_shader->local_ubo_size = 0;
    out_shader->local_ubo_stride = 0;
    out_shader->bound_instance_id = INVALID_ID;
    out_shader->attribute_stride = 0;

    // Setup arrays
    out_shader->global_texture_maps = darray_create(texture_map*);
    out_shader->uniforms = darray_create(shader_uniform);
    out_shader->attributes = darray_create(shader_attribute);

    // Create a hashtable to store uniform array indexes. This provides a direct index into the
    // 'uniforms' array stored in the shader for quick lookups by name.
    u64 element_size = sizeof(u16);  // Indexes are stored as u16s.
    u64 element_count = 1023;        // This is more uniforms than we will ever need, but a bigger table reduces collision chance.
    out_shader->hashtable_block = kallocate(element_size * element_count, MEMORY_TAG_HASHTABLE);
    hashtable_create(element_size, element_count, out_shader->hashtable_block, false, &out_shader->uniform_lookup);

    // Invalidate all spots in the hashtable.
    u32 invalid = INVALID_ID;
    hashtable_fill(&out_shader->uniform_lookup, &invalid);

    // A running total of the actual global uniform buffer object size.
    out_shader->global_ubo_size = 0;
    // A running total of the actual instance uniform buffer object size.
    out_shader->ubo_size = 0;
    // NOTE: UBO alignment requirement set in renderer backend.

    // This is hard-coded because the Vulkan spec only guarantees that a _minimum_ 128 bytes of space are available,
    // and it's up to the driver to determine how much is available. Therefore, to avoid complexity, only the
    // lowest common denominator of 128B will be used.
    // NOTE: This is used by the backend to setup the mapped_local_uniform_block.
    out_shader->local_ubo_stride = 128;

    // Take a copy of the flags.
    out_shader->flags = config->flags;
    out_shader->max_instances = config->max_instances;

    // Allocate and invalidate all instance states.
    out_shader->instance_states = kallocate(sizeof(shader_instance_state) * out_shader->max_instances, MEMORY_TAG_ARRAY);
    for (u32 i = 0; i < out_shader->max_instances; ++i) {
        out_shader->instance_states[i].id = INVALID_ID;
    }

    if (!renderer_shader_create(out_shader, config, pass)) {
        KERROR("Error creating shader.");
        return false;
    }

    // Ready to be initialized.
    out_shader->state = SHADER_STATE_UNINITIALIZED;

    // Process attributes
    for (u32 i = 0; i < config->attribute_count; ++i) {
        shader_attribute_config* ac = &config->attributes[i];
        if (!internal_attribute_add(out_shader, ac)) {
            KERROR("Failed to add attribute '%s' to shader '%s'.", ac->name, config->name);
            return false;
        }
    }

    // Process uniforms
    for (u32 i = 0; i < config->uniform_count; ++i) {
        shader_uniform_config* uc = &config->uniforms[i];
        if (uniform_type_is_sampler(uc->type)) {
            if (!internal_sampler_add(out_shader, uc)) {
                KERROR("Failed to add sampler '%s' to shader '%s'.", uc->name, config->name);
                return false;
            }
        } else {
            if (!internal_uniform_add(out_shader, uc, INVALID_ID)) {
                KERROR("Failed to add uniform '%s' to shader '%s'.", uc->name, config->name);
                return false;
            }
        }
    }

    // Initialize the shader.
    if (!renderer_shader_initialize(out_shader)) {
        KERROR("shader_system_create: initialization failed for shader '%s'.", config->name);
        // NOTE: initialize automatically destroys the shader if it fails.
        return false;
    }

    // Create the uniform buffer.
    u64 total_buffer_size = out_shader->global_ubo_stride + (out_shader->ubo_stride * out_shader->max_instances);
    char bufname[256];
    kzero_memory(bufname, 256);
    string_format(bufname, "renderbuffer_global_uniform");
    if (!renderer_renderbuffer_create(bufname, RENDERBUFFER_TYPE_UNIFORM, total_buffer_size, RENDERBUFFER_TRACK_TYPE_FREELIST, &out_shader->uniform_buffer)) {
        KERROR("Uniform buffer creation failed for object shader.");
        return false;
    }

    // Initialize will have figured out required stride, alignment, etc. by now, so
    // go ahead and bind and map the uniform buffer, then allocate space for the global UBO if
    // need be.
    renderer_renderbuffer_bind(&out_shader->uniform_buffer, 0);

    // Map the entire buffer's memory.
    out_shader->mapped_uniform_buffer_block = renderer_renderbuffer_map_memory(&out_shader->uniform_buffer, 0, -1);

    //  Allocate space for the global UBO, whcih should occupy the _stride_ space,
    //  _not_ the actual size used.
    if (out_shader->global_ubo_size > 0 && out_shader->global_ubo_stride > 0) {
        if (!renderer_renderbuffer_allocate(&out_shader->uniform_buffer, out_shader->global_ubo_stride, &out_shader->global_ubo_offset)) {
            KERROR("Failed to allocate space for the uniform buffer!");
            return false;
        }
    }

    // At this point, creation is successful, so store the shader id in the hashtable
    // so this can be looked up by name later.
    if (!hashtable_set(&state_ptr->lookup, config->name, &out_shader->id)) {
        // Dangit, we got so far... welp, nuke the shader and boot.
        renderer_shader_destroy(out_shader);
        return false;
    }

    return true;
}

u32 shader_system_get_id(const char* shader_name) {
    u32 shader_id = INVALID_ID;
    if (!hashtable_get(&state_ptr->lookup, shader_name, &shader_id)) {
        KERROR("There is no shader registered named '%s'.", shader_name);
        return INVALID_ID;
    }
    // KTRACE("Got id %u for shader named '%s'.", shader_id, shader_name);
    return shader_id;
}

shader* shader_system_get_by_id(u32 shader_id) {
    if (shader_id == INVALID_ID) {
        KERROR("shader_system_get_by_id was passed INVALID_ID. Null will be returned.");
        return 0;
    }
    if (state_ptr->shaders[shader_id].id == INVALID_ID) {
        KERROR("shader_system_get_by_id was passed an invalid id (%u. Null will be returned.", shader_id);
        return 0;
    }
    if (shader_id >= state_ptr->config.max_shader_count) {
        KERROR("shader_system_get_by_id was passed an id (%u) out of range (0-%u). Null will be returned.", shader_id, state_ptr->config.max_shader_count);
        return 0;
    }
    return &state_ptr->shaders[shader_id];
}

shader* shader_system_get(const char* shader_name) {
    u32 shader_id = shader_system_get_id(shader_name);
    if (shader_id != INVALID_ID) {
        return shader_system_get_by_id(shader_id);
    }
    return 0;
}

static void internal_shader_destroy(shader* s) {
    renderer_shader_destroy(s);

    // Set it to be unusable right away.
    s->state = SHADER_STATE_NOT_CREATED;

    u32 sampler_count = darray_length(s->global_texture_maps);
    for (u32 i = 0; i < sampler_count; ++i) {
        kfree(s->global_texture_maps[i], sizeof(texture_map), MEMORY_TAG_RENDERER);
    }
    darray_destroy(s->global_texture_maps);

    // Destroy uniform buffer.
    renderer_renderbuffer_destroy(&s->uniform_buffer);

    // Free the name.
    if (s->name) {
        u32 length = string_length(s->name);
        kfree(s->name, length + 1, MEMORY_TAG_STRING);
    }
    s->name = 0;
}

void shader_system_destroy(const char* shader_name) {
    u32 shader_id = shader_system_get_id(shader_name);
    if (shader_id == INVALID_ID) {
        return;
    }

    shader* s = &state_ptr->shaders[shader_id];

    // Release internal resources.
    internal_shader_destroy(s);

    // Nuke the instance states.
    kfree(s->instance_states, sizeof(shader_instance_state) * s->max_instances, MEMORY_TAG_ARRAY);
}

b8 shader_system_use(const char* shader_name) {
    u32 next_shader_id = shader_system_get_id(shader_name);
    if (next_shader_id == INVALID_ID) {
        return false;
    }

    return shader_system_use_by_id(next_shader_id);
}

b8 shader_system_use_by_id(u32 shader_id) {
    // Only perform the use if the shader id is different.
    // if (state_ptr->current_shader_id != shader_id) {
    shader* next_shader = shader_system_get_by_id(shader_id);
    state_ptr->current_shader_id = shader_id;
    if (!renderer_shader_use(next_shader)) {
        KERROR("Failed to use shader '%s'.", next_shader->name);
        return false;
    }
    if (!renderer_shader_bind_globals(next_shader)) {
        KERROR("Failed to bind globals for shader '%s'.", next_shader->name);
        return false;
    }
    // }
    return true;
}

u16 shader_system_uniform_location(shader* s, const char* uniform_name) {
    if (!s || s->id == INVALID_ID) {
        KERROR("shader_system_uniform_location called with invalid shader.");
        return INVALID_ID_U16;
    }

    u16 index = INVALID_ID_U16;
    if (!hashtable_get(&s->uniform_lookup, uniform_name, &index) || index == INVALID_ID_U16) {
        KERROR("Shader '%s' does not have a registered uniform named '%s'", s->name, uniform_name);
        return INVALID_ID_U16;
    }
    return s->uniforms[index].index;
}

b8 shader_system_uniform_set(const char* uniform_name, const void* value) {
    return shader_system_uniform_set_arrayed(uniform_name, 0, value);
}

b8 shader_system_uniform_set_arrayed(const char* uniform_name, u32 array_index, const void* value) {
    if (state_ptr->current_shader_id == INVALID_ID) {
        KERROR("shader_system_uniform_set called without a shader in use.");
        return false;
    }
    shader* s = &state_ptr->shaders[state_ptr->current_shader_id];
    u16 index = shader_system_uniform_location(s, uniform_name);
    return shader_system_uniform_set_by_location_arrayed(index, array_index, value);
}

b8 shader_system_sampler_set(const char* sampler_name, const texture* t) {
    return shader_system_sampler_set_arrayed(sampler_name, 0, t);
}

b8 shader_system_sampler_set_arrayed(const char* sampler_name, u32 array_index, const texture* t) {
    return shader_system_uniform_set_arrayed(sampler_name, array_index, t);
}

b8 shader_system_sampler_set_by_location(u16 location, const texture* t) {
    return shader_system_uniform_set_by_location_arrayed(location, 0, t);
}

b8 shader_system_uniform_set_by_location(u16 location, const void* value) {
    return shader_system_uniform_set_by_location_arrayed(location, 0, value);
}

b8 shader_system_uniform_set_by_location_arrayed(u16 location, u32 array_index, const void* value) {
    shader* s = &state_ptr->shaders[state_ptr->current_shader_id];
    shader_uniform* uniform = &s->uniforms[location];

    // Automatically bind scope if it's wrong.
    if (s->bound_scope != uniform->scope) {
        if (uniform->scope == SHADER_SCOPE_GLOBAL) {
            renderer_shader_bind_globals(s);
        } else if (uniform->scope == SHADER_SCOPE_INSTANCE) {
            renderer_shader_bind_instance(s, s->bound_instance_id);
        } else {
            renderer_shader_bind_local(s);
        }
        s->bound_scope = uniform->scope;
    }

    shader_instance_state* instance_state = &s->instance_states[s->bound_instance_id];

    b8 is_sampler = uniform_type_is_sampler(uniform->type);
    if (is_sampler) {
        texture_map* map = (texture_map*)value;
        if (uniform->scope == SHADER_SCOPE_GLOBAL) {
            return sampler_state_try_set(s->global_sampler_uniforms, s->global_uniform_sampler_count, uniform->location, array_index, map);
        } else if (uniform->scope == SHADER_SCOPE_INSTANCE) {
            return sampler_state_try_set(instance_state->sampler_uniforms, s->instance_uniform_sampler_count, uniform->location, array_index, map);
        } else {
            KERROR("Samplers may not be bound at the local level.");
            return false;
        }
    } else {
        // Obtain the address of the appropriate buffer, and the position within it.
        u64 addr;
        if (uniform->scope == SHADER_SCOPE_LOCAL) {
            addr = (u64)s->mapped_local_uniform_block;
            addr += uniform->offset + (uniform->size * array_index);
        } else {
            addr = (u64)s->mapped_uniform_buffer_block;
            addr += s->bound_ubo_offset + uniform->offset + (uniform->size * array_index);
        }

        // Copy the data over.
        kcopy_memory((void*)addr, value, uniform->size);
    }

    // Handle anything required in the renderer backend.
    return renderer_shader_uniform_set(s, uniform, array_index, value);
}

b8 shader_system_apply_global(b8 needs_update, frame_data* p_frame_data) {
    return renderer_shader_apply_globals(&state_ptr->shaders[state_ptr->current_shader_id], needs_update, p_frame_data);
}

b8 shader_system_bind_global(void) {
    shader* s = &state_ptr->shaders[state_ptr->current_shader_id];
    s->bound_scope = SHADER_SCOPE_GLOBAL;
    s->bound_instance_id = INVALID_ID;
    s->bound_ubo_offset = s->global_ubo_offset;
    return renderer_shader_bind_globals(s);
}

b8 shader_system_apply_instance(b8 needs_update, frame_data* p_frame_data) {
    return renderer_shader_apply_instance(&state_ptr->shaders[state_ptr->current_shader_id], needs_update, p_frame_data);
}

b8 shader_system_bind_instance(u32 instance_id) {
    if (instance_id == INVALID_ID) {
        KERROR("Cannot bind instance INVALID_ID.");
        return false;
    }

    // Set bind scope and bound instance id.
    shader* s = &state_ptr->shaders[state_ptr->current_shader_id];
    s->bound_scope = SHADER_SCOPE_INSTANCE;
    s->bound_instance_id = instance_id;

    // Set the bound ubo offset.
    shader_instance_state* instance_state = &s->instance_states[instance_id];
    s->bound_ubo_offset = instance_state->offset;
    return renderer_shader_bind_instance(s, instance_id);
}

b8 shader_system_apply_local(struct frame_data* p_frame_data) {
    shader* s = &state_ptr->shaders[state_ptr->current_shader_id];
    return renderer_shader_apply_local(s, p_frame_data);
}

b8 shader_system_bind_local(void) {
    shader* s = &state_ptr->shaders[state_ptr->current_shader_id];
    s->bound_scope = SHADER_SCOPE_LOCAL;
    s->bound_instance_id = INVALID_ID;
    return renderer_shader_bind_local(s);
}

b8 shader_system_instance_resources_acquire(shader* s, const shader_instance_resource_config* config, u32* out_instance_id) {
    if (!s) {
        KERROR("shader_system_instance_resources_acquire requires a valid pointer to a shader.");
        return false;
    }
    if (!config) {
        KERROR("shader_system_instance_resources_acquire requires a valid pointer to a configuration.");
        return false;
    }
    if (!out_instance_id) {
        KERROR("shader_system_instance_resources_acquire requires a valid pointer to hold a new instance id.");
        return false;
    }

    u32 instance_id = INVALID_ID;
    for (u32 i = 0; i < s->max_instances; ++i) {
        if (s->instance_states[i].id == INVALID_ID) {
            s->instance_states[i].id = i;
            instance_id = i;
            break;
        }
    }
    if (instance_id == INVALID_ID) {
        KERROR("shader_system_acquire_instance_resources failed to acquire new id for shader '%s', max instances=%u", s->name, s->max_instances);
        return false;
    }

    texture* default_texture = texture_system_get_default_texture();

    // Map texture maps in the config to the correct uniforms
    shader_instance_state* instance_state = &s->instance_states[instance_id];
    // Only setup if the shader actually requires it.
    if (s->instance_texture_count > 0) {
        instance_state->sampler_uniforms = kallocate(sizeof(shader_uniform_sampler_state) * s->instance_uniform_sampler_count, MEMORY_TAG_ARRAY);

        // Assign uniforms to each of the sampler states.
        for (u32 ii = 0; ii < s->instance_uniform_sampler_count; ++ii) {
            shader_uniform_sampler_state* sampler_state = &instance_state->sampler_uniforms[ii];
            sampler_state->uniform = &s->uniforms[s->instance_sampler_indices[ii]];

            // Grab the uniform texture config as well.
            shader_instance_uniform_texture_config* tc = &config->uniform_configs[ii];

            u32 array_length = KMAX(sampler_state->uniform->array_length, 1);
            // Setup the array for the sampler texture maps.
            sampler_state->uniform_texture_maps = kallocate(sizeof(texture_map*) * array_length, MEMORY_TAG_ARRAY);
            // Setup descriptor states
            sampler_state->descriptor_states = kallocate(sizeof(shader_descriptor_state) * array_length, MEMORY_TAG_ARRAY);
            // Per descriptor
            for (u32 d = 0; d < array_length; ++d) {
                sampler_state->uniform_texture_maps[d] = tc->texture_maps[d];
                // Make sure it has a texture map assigned. Use default if not.
                if (!sampler_state->uniform_texture_maps[d]->texture) {
                    sampler_state->uniform_texture_maps[d]->texture = default_texture;
                }
                // Per frame
                // TODO: handle different frame counts.
                for (u32 j = 0; j < 3; ++j) {
                    sampler_state->descriptor_states[d].generations[j] = INVALID_ID_U8;
                    sampler_state->descriptor_states[d].ids[j] = INVALID_ID;
                }
            }
        }
    }

    // Allocate some space in the UBO - by the stride, not the size.
    u64 size = s->ubo_stride;
    if (size > 0) {
        if (!renderer_renderbuffer_allocate(&s->uniform_buffer, size, &instance_state->offset)) {
            KERROR("vulkan_material_shader_acquire_resources failed to acquire ubo space");
            return false;
        }
    }

    // UBO binding. NOTE: really only matters where there are instance uniforms, but set them anyway.
    // TODO: Handle different frame counts.
    for (u32 j = 0; j < 3; ++j) {
        instance_state->ubo_descriptor_state.generations[j] = INVALID_ID_U8;
        instance_state->ubo_descriptor_state.ids[j] = INVALID_ID_U8;
    }

    *out_instance_id = instance_id;

    // Do the renderer-specific things.
    return renderer_shader_instance_resources_acquire(s, instance_id);
}

b8 shader_system_instance_resources_release(shader* s, u32 instance_id) {
    if (!s) {
        KERROR("shader_system_instance_resources_release requires a valid pointer to a shader.");
        return false;
    }
    if (instance_id == INVALID_ID) {
        KERROR("shader_system_instance_resources_releases cannot have an instance id of INVALID_ID.");
        return false;
    }
    shader_instance_state* instance_state = &s->instance_states[instance_id];
    // Invalidate UBO descriptor state.
    for (u32 j = 0; j < 3; ++j) {
        instance_state->ubo_descriptor_state.generations[j] = INVALID_ID_U8;
        instance_state->ubo_descriptor_state.ids[j] = INVALID_ID_U8;
    }

    // Destroy bindings and their descriptor states/uniforms.
    for (u32 a = 0; a < s->instance_uniform_sampler_count; ++a) {
        shader_uniform_sampler_state* sampler_state = &instance_state->sampler_uniforms[a];
        u32 array_length = KMAX(sampler_state->uniform->array_length, 1);
        kfree(sampler_state->descriptor_states, sizeof(shader_descriptor_state) * array_length, MEMORY_TAG_ARRAY);
        sampler_state->descriptor_states = 0;
        kfree(sampler_state->uniform_texture_maps, sizeof(texture_map*) * array_length, MEMORY_TAG_ARRAY);
        sampler_state->uniform_texture_maps = 0;
    }

    if (s->ubo_stride != 0) {
        if (!renderer_renderbuffer_free(&s->uniform_buffer, s->ubo_stride, instance_state->offset)) {
            KERROR("vulkan_renderer_shader_release_instance_resources failed to free range from renderbuffer.");
        }
    }
    instance_state->offset = INVALID_ID;
    instance_state->id = INVALID_ID;

    // Do the renderer-specific things.
    return renderer_shader_instance_resources_release(s, instance_id);
}

static b8 sampler_state_try_set(shader_uniform_sampler_state* sampler_uniforms, u32 sampler_count, u16 uniform_location, u32 array_index, texture_map* map) {
    // Find the sampler uniform state to update.
    for (u32 i = 0; i < sampler_count; ++i) {
        shader_uniform_sampler_state* su = &sampler_uniforms[i];
        if (su->uniform->location == uniform_location) {
            if (su->uniform->array_length > 1) {
                if (array_index >= su->uniform->array_length) {
                    KERROR("uniform_set error: array_index (%u) is out of range (0-%u)", array_index, su->uniform->array_length);
                    return false;
                }
                su->uniform_texture_maps[array_index] = map;
            } else {
                su->uniform_texture_maps[0] = map;
            }
            return true;
        }
    }
    KERROR("sampler_state_try_set: Unable to find uniform location %u. Sampler uniform not set.", uniform_location);
    return false;
}

static b8 internal_attribute_add(shader* shader, const shader_attribute_config* config) {
    u32 size = 0;
    switch (config->type) {
        case SHADER_ATTRIB_TYPE_INT8:
        case SHADER_ATTRIB_TYPE_UINT8:
            size = 1;
            break;
        case SHADER_ATTRIB_TYPE_INT16:
        case SHADER_ATTRIB_TYPE_UINT16:
            size = 2;
            break;
        case SHADER_ATTRIB_TYPE_FLOAT32:
        case SHADER_ATTRIB_TYPE_INT32:
        case SHADER_ATTRIB_TYPE_UINT32:
            size = 4;
            break;
        case SHADER_ATTRIB_TYPE_FLOAT32_2:
            size = 8;
            break;
        case SHADER_ATTRIB_TYPE_FLOAT32_3:
            size = 12;
            break;
        case SHADER_ATTRIB_TYPE_FLOAT32_4:
            size = 16;
            break;
        default:
            KERROR("Unrecognized type %d, defaulting to size of 4. This probably is not what is desired.");
            size = 4;
            break;
    }

    shader->attribute_stride += size;

    // Create/push the attribute.
    shader_attribute attrib = {};
    attrib.name = string_duplicate(config->name);
    attrib.size = size;
    attrib.type = config->type;
    darray_push(shader->attributes, attrib);

    return true;
}

static b8 internal_sampler_add(shader* shader, shader_uniform_config* config) {
    // Samples can't be used for push constants.
    if (config->scope == SHADER_SCOPE_LOCAL) {
        KERROR("add_sampler cannot add a sampler at local scope.");
        return false;
    }

    // Verify the name is valid and unique.
    if (!uniform_name_valid(shader, config->name) || !shader_uniform_add_state_valid(shader)) {
        return false;
    }

    // If global, push into the global list.
    u32 location = 0;
    if (config->scope == SHADER_SCOPE_GLOBAL) {
        u32 global_texture_count = darray_length(shader->global_texture_maps);
        if (global_texture_count + 1 > state_ptr->config.max_global_textures) {
            KERROR("Shader global texture count %i exceeds max of %i", global_texture_count, state_ptr->config.max_global_textures);
            return false;
        }
        location = global_texture_count;

        // NOTE: creating a default texture map to be used here. Can always be updated later.
        texture_map default_map = {};
        default_map.filter_magnify = TEXTURE_FILTER_MODE_LINEAR;
        default_map.filter_minify = TEXTURE_FILTER_MODE_LINEAR;
        default_map.repeat_u = default_map.repeat_v = default_map.repeat_w = TEXTURE_REPEAT_REPEAT;

        // Allocate a pointer assign the texture, and push into global texture maps.
        // NOTE: This allocation is only done for global texture maps.
        texture_map* map = kallocate(sizeof(texture_map), MEMORY_TAG_RENDERER);
        *map = default_map;
        map->texture = texture_system_get_default_texture();

        if (!renderer_texture_map_resources_acquire(map)) {
            KERROR("Failed to acquire resources for global texture map during shader creation.");
            return false;
        }

        darray_push(shader->global_texture_maps, map);
    } else {
        // Otherwise, it's instance-level, so keep count of how many need to be added during the resource acquisition.
        if (shader->instance_texture_count + 1 > state_ptr->config.max_instance_textures) {
            KERROR("Shader instance texture count %i exceeds max of %i", shader->instance_texture_count, state_ptr->config.max_instance_textures);
            return false;
        }
        location = shader->instance_texture_count;
        shader->instance_texture_count++;
    }

    // Treat it like a uniform. NOTE: In the case of samplers, out_location is used to determine the
    // hashtable entry's 'location' field value directly, and is then set to the index of the uniform array.
    // This allows location lookups for samplers as if they were uniforms as well (since technically they are).
    // TODO: might need to store this elsewhere
    if (!internal_uniform_add(shader, config, location)) {
        KERROR("Unable to add sampler uniform.");
        return false;
    }

    return true;
}

static u32 generate_new_shader_id(void) {
    for (u32 i = 0; i < state_ptr->config.max_shader_count; ++i) {
        if (state_ptr->shaders[i].id == INVALID_ID) {
            return i;
        }
    }
    return INVALID_ID;
}

static b8 internal_uniform_add(shader* shader, const shader_uniform_config* config, u32 location) {
    if (!shader_uniform_add_state_valid(shader) || !uniform_name_valid(shader, config->name)) {
        return false;
    }
    u32 uniform_count = darray_length(shader->uniforms);
    if (uniform_count + 1 > state_ptr->config.max_uniform_count) {
        KERROR("A shader can only accept a combined maximum of %d uniforms and samplers at global, instance and local scopes.", state_ptr->config.max_uniform_count);
        return false;
    }
    b8 is_sampler = uniform_type_is_sampler(config->type);
    shader_uniform entry;
    entry.index = uniform_count;  // Index is saved to the hashtable for lookups.
    entry.scope = config->scope;
    entry.type = config->type;
    entry.array_length = config->array_length;
    b8 is_global = (config->scope == SHADER_SCOPE_GLOBAL);
    if (is_sampler) {
        // Just use the passed in location
        entry.location = location;
    } else {
        entry.location = entry.index;
    }

    if (config->scope == SHADER_SCOPE_LOCAL) {
        entry.set_index = 2;  // NOTE: set 2 doesn't exist in Vulkan, it's a push constant.
        entry.offset = shader->local_ubo_size;
        entry.size = config->size;
    } else {
        entry.set_index = (u32)config->scope;
        entry.offset = is_sampler ? 0 : is_global ? shader->global_ubo_size
                                                  : shader->ubo_size;
        entry.size = is_sampler ? 0 : config->size;
    }

    if (!hashtable_set(&shader->uniform_lookup, config->name, &entry.index)) {
        KERROR("Failed to add uniform.");
        return false;
    }
    darray_push(shader->uniforms, entry);

    if (!is_sampler) {
        if (entry.scope == SHADER_SCOPE_GLOBAL) {
            shader->global_ubo_size += (entry.size * entry.array_length);
        } else if (entry.scope == SHADER_SCOPE_INSTANCE) {
            shader->ubo_size += (entry.size * entry.array_length);
        } else if (entry.scope == SHADER_SCOPE_LOCAL) {
            shader->local_ubo_size += (entry.size * entry.array_length);
        }
    }

    return true;
}

static b8 uniform_name_valid(shader* shader, const char* uniform_name) {
    if (!uniform_name || !string_length(uniform_name)) {
        KERROR("Uniform name must exist.");
        return false;
    }
    u16 location;
    if (hashtable_get(&shader->uniform_lookup, uniform_name, &location) && location != INVALID_ID_U16) {
        KERROR("A uniform by the name '%s' already exists on shader '%s'.", uniform_name, shader->name);
        return false;
    }
    return true;
}

static b8 shader_uniform_add_state_valid(shader* shader) {
    if (shader->state != SHADER_STATE_UNINITIALIZED) {
        KERROR("Uniforms may only be added to shaders before initialization.");
        return false;
    }
    return true;
}
