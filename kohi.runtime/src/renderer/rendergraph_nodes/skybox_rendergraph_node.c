#include "skybox_rendergraph_node.h"

#include "core/engine.h"
#include "identifiers/khandle.h"
#include "logger.h"
#include "memory/kmemory.h"
#include "renderer/renderer_frontend.h"
#include "renderer/renderer_types.h"
#include "renderer/rendergraph.h"
#include "resources/skybox.h"
#include "strings/kstring.h"
#include "systems/resource_system.h"
#include "systems/shader_system.h"

typedef struct skybox_shader_locations {
    u16 projection_location;
    u16 view_location;
    u16 cube_map_location;
} skybox_shader_locations;

typedef struct skybox_renderpass_node_internal_data {
    struct renderer_system_state* renderer;

    shader* s;
    skybox_shader_locations locations;

    // The internal renderer renderpass.
    renderpass internal_pass;

    k_handle colourbuffer_handle;

    skybox* sb;

    struct viewport* vp;
    mat4 view;
    mat4 projection;
} skybox_rendergraph_node_internal_data;

KAPI b8 skybox_rendergraph_node_create(struct rendergraph* graph, struct rendergraph_node* self, const struct rendergraph_node_config* config) {
    if (!self) {
        KERROR("skybox_rendergraph_node_create requires a valid pointer to a pass");
        return false;
    }
    if (!config) {
        KERROR("skybox_rendergraph_node_create requires a valid configuration.");
        return false;
    }

    // Setup internal data.
    self->internal_data = kallocate(sizeof(skybox_rendergraph_node_internal_data), MEMORY_TAG_RENDERER);
    skybox_rendergraph_node_internal_data* internal_data = self->internal_data;
    internal_data->renderer = engine_systems_get()->renderer_system;

    self->name = string_duplicate(config->name);

    // Has one sink, for the colourbuffer.
    self->sink_count = 1;
    self->sinks = kallocate(sizeof(rendergraph_sink) * self->sink_count, MEMORY_TAG_ARRAY);

    rendergraph_node_sink_config* sink_config = 0;
    for (u32 i = 0; i < config->sink_count; ++i) {
        rendergraph_node_sink_config* sink = &config->sinks[i];
        if (strings_equali("colourbuffer", sink->name)) {
            sink_config = sink;
            break;
        }
    }

    if (sink_config) {
        // Setup the colourbuffer sink.
        rendergraph_sink* colourbuffer_sink = &self->sinks[0];
        colourbuffer_sink->name = string_duplicate("colourbuffer");
        colourbuffer_sink->type = RENDERGRAPH_RESOURCE_TYPE_FRAMEBUFFER;
        colourbuffer_sink->bound_source = 0;
        // Save off the configured source name for later lookup and binding.
        colourbuffer_sink->configured_source_name = string_duplicate(sink_config->source_name);
    } else {
        KERROR("Skybox rendergraph node requires configuration for sink called 'colourbuffer'.");
        return false;
    }

    // Has one source, for the colourbuffer.
    self->source_count = 1;
    self->sources = kallocate(sizeof(rendergraph_source) * self->source_count, MEMORY_TAG_ARRAY);

    // Setup the colourbuffer source.
    rendergraph_source* colourbuffer_source = &self->sources[0];
    colourbuffer_source->name = string_duplicate("colourbuffer");
    colourbuffer_source->type = RENDERGRAPH_RESOURCE_TYPE_FRAMEBUFFER;
    colourbuffer_source->value.framebuffer_handle = k_handle_invalid();
    colourbuffer_source->is_bound = false;

    // Function pointers.
    self->initialize = skybox_rendergraph_node_initialize;
    self->destroy = skybox_rendergraph_node_destroy;
    self->load_resources = skybox_rendergraph_node_load_resources;
    self->execute = skybox_rendergraph_node_execute;

    return true;
}

b8 skybox_rendergraph_node_initialize(struct rendergraph_node* self) {
    if (!self) {
        return false;
    }

    skybox_rendergraph_node_internal_data* internal_data = self->internal_data;

    // Setup the renderpass.
    renderpass_config skybox_rendergraph_node_config = {0};
    skybox_rendergraph_node_config.name = "Renderpass.Skybox";
    skybox_rendergraph_node_config.attachment_count = 1;
    // TODO: leaking this?
    skybox_rendergraph_node_config.attachment_configs = kallocate(sizeof(renderpass_attachment_config) * skybox_rendergraph_node_config.attachment_count, MEMORY_TAG_ARRAY);

    // Color attachment.
    renderpass_attachment_config* skybox_target_colour = &skybox_rendergraph_node_config.attachment_configs[0];
    skybox_target_colour->type = RENDERER_ATTACHMENT_TYPE_FLAG_COLOUR_BIT;
    skybox_target_colour->load_op = RENDERER_ATTACHMENT_LOAD_OPERATION_DONT_CARE;

    // Only need to store if something is bound to the output source.
    if (self->sources[0].is_bound) {
        skybox_target_colour->store_op = RENDERER_ATTACHMENT_STORE_OPERATION_STORE;
        skybox_target_colour->post_pass_use = RENDERER_ATTACHMENT_USE_COLOUR_ATTACHMENT;
    } else {
        skybox_target_colour->store_op = RENDERER_ATTACHMENT_STORE_OPERATION_DONT_CARE;
    }

    if (!renderer_renderpass_create(&skybox_rendergraph_node_config, &internal_data->internal_pass)) {
        KERROR("Skybox rendergraph pass - Failed to create skybox renderpass ");
        return false;
    }

    // Load skybox shader.
    const char* skybox_shader_name = "Shader.Builtin.Skybox";
    resource skybox_shader_config_resource;
    if (!resource_system_load(skybox_shader_name, RESOURCE_TYPE_SHADER, 0, &skybox_shader_config_resource)) {
        KERROR("Failed to load skybox shader resource.");
        return false;
    }
    shader_config* skybox_shader_config = (shader_config*)skybox_shader_config_resource.data;
    if (!shader_system_create(&internal_data->internal_pass, skybox_shader_config)) {
        KERROR("Failed to create skybox shader.");
        return false;
    }

    resource_system_unload(&skybox_shader_config_resource);
    // Get a pointer to the shader.
    internal_data->s = shader_system_get(skybox_shader_name);
    internal_data->locations.projection_location = shader_system_uniform_location(internal_data->s, "projection");
    internal_data->locations.view_location = shader_system_uniform_location(internal_data->s, "view");
    internal_data->locations.cube_map_location = shader_system_uniform_location(internal_data->s, "cube_texture");

    return true;
}

b8 skybox_rendergraph_node_load_resources(struct rendergraph_node* self) {
    if (!self) {
        return false;
    }

    // Resolve framebuffer handle via bound source.
    skybox_rendergraph_node_internal_data* internal_data = self->internal_data;
    if (self->sinks[0].bound_source) {
        internal_data->colourbuffer_handle = self->sinks[0].bound_source->value.framebuffer_handle;
        return true;
    }

    return false;
}

b8 skybox_rendergraph_node_execute(struct rendergraph_node* self, struct frame_data* p_frame_data) {
    if (!self) {
        return false;
    }

    skybox_rendergraph_node_internal_data* internal_data = self->internal_data;

    // Bind the viewport
    renderer_active_viewport_set(internal_data->vp);

    if (!renderer_renderpass_begin(&internal_data->internal_pass, internal_data->colourbuffer_handle)) {
        KERROR("skybox pass failed to start.");
        return false;
    }

    if (internal_data->sb) {
        shader_system_use_by_id(internal_data->s->id);

        // Get the view matrix, but zero out the position so the skybox stays put on screen.
        mat4 view_matrix = internal_data->view;
        view_matrix.data[12] = 0.0f;
        view_matrix.data[13] = 0.0f;
        view_matrix.data[14] = 0.0f;

        // Apply globals
        renderer_shader_bind_globals(internal_data->s);
        if (!shader_system_uniform_set_by_location(internal_data->locations.projection_location, &internal_data->projection)) {
            KERROR("Failed to apply skybox projection uniform.");
            return false;
        }
        if (!shader_system_uniform_set_by_location(internal_data->locations.view_location, &view_matrix)) {
            KERROR("Failed to apply skybox view uniform.");
            return false;
        }
        shader_system_apply_global(true, p_frame_data);

        // Instance
        shader_system_bind_instance(internal_data->sb->instance_id);
        if (!shader_system_uniform_set_by_location(internal_data->locations.cube_map_location, &internal_data->sb->cubemap)) {
            KERROR("Failed to apply skybox cube map uniform.");
            return false;
        }

        // FIXME: This likely will break things.
        b8 needs_update = true; // ext_data->sb->render_frame_number != p_frame_data->renderer_frame_number || ext_data->sb->draw_index != p_frame_data->draw_index;
        shader_system_apply_instance(needs_update, p_frame_data);

        // Draw it.
        geometry_render_data render_data = {};
        render_data.material = internal_data->sb->g->material;
        render_data.vertex_count = internal_data->sb->g->vertex_count;
        render_data.vertex_element_size = internal_data->sb->g->vertex_element_size;
        render_data.vertex_buffer_offset = internal_data->sb->g->vertex_buffer_offset;
        render_data.index_count = internal_data->sb->g->index_count;
        render_data.index_element_size = internal_data->sb->g->index_element_size;
        render_data.index_buffer_offset = internal_data->sb->g->index_buffer_offset;

        renderer_geometry_draw(&render_data);
    }

    if (!renderer_renderpass_end(&internal_data->internal_pass)) {
        KERROR("skybox pass failed to end.");
        return false;
    }

    return true;
}

void skybox_rendergraph_node_destroy(struct rendergraph_node* self) {
    if (self) {
        if (self->internal_data) {
            skybox_rendergraph_node_internal_data* internal_data = self->internal_data;
            // Destroy the pass.
            renderer_renderpass_destroy(&internal_data->internal_pass);
            kfree(self->internal_data, sizeof(skybox_rendergraph_node_internal_data), MEMORY_TAG_RENDERER);
            self->internal_data = 0;
        }
    }
}

void skybox_rendergraph_node_set_skybox(struct rendergraph_node* self, struct skybox* sb) {
    if (self) {
        skybox_rendergraph_node_internal_data* internal_data = self->internal_data;
        internal_data->sb = sb;
    }
}

void skybox_rendergraph_node_set_viewport_and_matrices(struct rendergraph_node* self, struct viewport* vp, mat4 view, mat4 projection) {
    if (self) {
        if (self->internal_data) {
            skybox_rendergraph_node_internal_data* internal_data = self->internal_data;
            internal_data->vp = vp;
            internal_data->view = view;
            internal_data->projection = projection;
        }
    }
}

b8 skybox_rendergraph_node_register_factory(void) {
    rendergraph_node_factory factory = {0};
    factory.type = "skybox";
    factory.create = skybox_rendergraph_node_create;
    return rendergraph_system_node_factory_register(engine_systems_get()->rendergraph_system, &factory);
}