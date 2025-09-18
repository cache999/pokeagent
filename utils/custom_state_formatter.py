from world.mapper import Map

def _format_saved_map_info(map_info, saved_map, include_debug_info=False, include_npcs=True):
    """Format map and traversability information, combining world.mapper.Map
    and currently observed state data.
    """

    context_parts = []

    if not map_info:
        return context_parts

    context_parts.append("\n=== LOCATION & MAP INFO ===")

    if 'current_map' in map_info:
        context_parts.append(f"Current Map: {map_info['current_map']}")

    # Use raw tiles if available
    if 'tiles' in map_info and map_info['tiles']:
        raw_tiles = map_info['tiles']
        # Get player facing direction from context
        facing = "South"  # default
        try:
            # Try to get facing from state data
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'state_data' in frame.f_locals:
                    facing = frame.f_locals['state_data'].get('player', {}).get('facing', 'South')
                    break
                frame = frame.f_back
        except:
            pass

        # Get NPCs from map info
        npcs = map_info.get('object_events', []) if include_npcs else []

        # Get player coordinates for NPC positioning
        player_coords = map_info.get('player_coords')

        # Use unified LLM formatter for consistency
        map_display = format_map_for_llm(raw_tiles, facing, npcs, player_coords)
        context_parts.append(f"\n--- FULL TRAVERSABILITY MAP ({len(raw_tiles)}x{len(raw_tiles[0])}) ---")
        context_parts.append(map_display)

        # Add dynamic legend based on symbols in the map
        grid = format_map_grid(raw_tiles, facing, npcs, player_coords)
        legend = generate_dynamic_legend(grid)
        context_parts.append(f"\n{legend}")

        # Add NPC information if present
        if include_npcs and npcs:
            context_parts.append(f"\n--- NPCs/TRAINERS ({len(npcs)} found) ---")
            context_parts.append(
                "NOTE: These are static NPC spawn positions. NPCs may have moved from these locations during walking animations.")

            # Analyze terrain under NPCs
            for npc in npcs:
                npc_x = npc.get('current_x', 0)
                npc_y = npc.get('current_y', 0)
                npc_info = f"NPC {npc['id']}: "

                if npc.get('trainer_type', 0) > 0:
                    npc_info += f"Trainer at ({npc_x}, {npc_y})"
                else:
                    npc_info += f"NPC at ({npc_x}, {npc_y})"

                # Analyze terrain under NPC position
                terrain_note = _analyze_npc_terrain(npc, raw_tiles, player_coords)
                if terrain_note:
                    npc_info += f" - {terrain_note}"

                context_parts.append(npc_info)

    return context_parts