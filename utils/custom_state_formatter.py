from utils import map_formatter
from utils.map_formatter import format_tile_to_symbol
from utils.state_formatter import _get_player_position, _format_party_info, _format_game_state, _format_debug_info, \
    _analyze_npc_terrain
from world.mapper import World


def custom_format_state_for_llm(state_data, world: World = None, include_debug_info=False, include_npcs=True):
    """
    Custom state formatter that uses saved_map for map rendering.
    """
    return _custom_format_state_detailed(state_data, world, include_debug_info, include_npcs)


def _custom_format_state_detailed(state_data, world: World, include_debug_info=False, include_npcs=True):
    """
    Internal function to create detailed multi-line state format for LLM prompts.


    Coordinate systems:
    - Player position
    - Saved map position
    - View coordinates (always start at 0,0 in top-left of view) - this is what's passed to the LLM


    """
    context_parts = []

    # Check both player and game sections for data
    player_data = state_data.get('player', {})
    game_data = state_data.get('game', {})

    # Check if we're in battle to determine formatting mode
    is_in_battle = game_data.get('is_in_battle', False) or game_data.get('in_battle', False)

    if is_in_battle:
        # BATTLE MODE: Focus on battle-relevant information
        context_parts.append("=== BATTLE MODE ===")
        context_parts.append("Currently in battle - map and dialogue information hidden")

        # Battle information first
        if 'battle_info' in game_data and game_data['battle_info']:
            battle = game_data['battle_info']
            context_parts.append("\n=== BATTLE STATUS ===")

            # Battle type and context
            battle_type = battle.get('battle_type', 'unknown')
            context_parts.append(f"Battle Type: {battle_type.title()}")
            if battle.get('is_capturable'):
                context_parts.append("🟢 Wild Pokémon - CAN BE CAPTURED")
            if battle.get('can_escape'):
                context_parts.append("🟡 Can escape from battle")

            # Player's active Pokémon
            if 'player_pokemon' in battle and battle['player_pokemon']:
                player_pkmn = battle['player_pokemon']
                context_parts.append(f"\n--- YOUR POKÉMON ---")
                context_parts.append(
                    f"{player_pkmn.get('nickname', player_pkmn.get('species', 'Unknown'))} (Lv.{player_pkmn.get('level', '?')})")

                # Health display with percentage
                current_hp = player_pkmn.get('current_hp', 0)
                max_hp = player_pkmn.get('max_hp', 1)
                hp_pct = player_pkmn.get('hp_percentage', 0)
                health_bar = "🟢" if hp_pct > 50 else "🟡" if hp_pct > 25 else "🔴"
                context_parts.append(f"  HP: {current_hp}/{max_hp} ({hp_pct}%) {health_bar}")

                # Status condition
                status = player_pkmn.get('status', 'Normal')
                if status != 'Normal':
                    context_parts.append(f"  Status: {status}")

                # Types
                types = player_pkmn.get('types', [])
                if types:
                    context_parts.append(f"  Type: {'/'.join(types)}")

                # Available moves with PP
                moves = player_pkmn.get('moves', [])
                move_pp = player_pkmn.get('move_pp', [])
                if moves:
                    context_parts.append(f"  Moves:")
                    for i, move in enumerate(moves):
                        if move and move.strip():
                            pp = move_pp[i] if i < len(move_pp) else '?'
                            context_parts.append(f"    {i + 1}. {move} (PP: {pp})")

            # Opponent Pokémon
            if 'opponent_pokemon' in battle:
                if battle['opponent_pokemon']:
                    opp_pkmn = battle['opponent_pokemon']
                    context_parts.append(f"\n--- OPPONENT POKÉMON ---")
                    context_parts.append(f"{opp_pkmn.get('species', 'Unknown')} (Lv.{opp_pkmn.get('level', '?')})")

                    # Health display with percentage
                    current_hp = opp_pkmn.get('current_hp', 0)
                    max_hp = opp_pkmn.get('max_hp', 1)
                    hp_pct = opp_pkmn.get('hp_percentage', 0)
                    health_bar = "🟢" if hp_pct > 50 else "🟡" if hp_pct > 25 else "🔴"
                    context_parts.append(f"  HP: {current_hp}/{max_hp} ({hp_pct}%) {health_bar}")

                    # Status condition
                    status = opp_pkmn.get('status', 'Normal')
                    if status != 'Normal':
                        context_parts.append(f"  Status: {status}")

                    # Types
                    types = opp_pkmn.get('types', [])
                    if types:
                        context_parts.append(f"  Type: {'/'.join(types)}")

                    # Moves (for wild Pokémon, showing moves can help with strategy)
                    moves = opp_pkmn.get('moves', [])
                    if moves and any(move.strip() for move in moves):
                        context_parts.append(f"  Known Moves:")
                        for i, move in enumerate(moves):
                            if move and move.strip():
                                context_parts.append(f"    • {move}")

                    # Stats (helpful for battle strategy)
                    stats = opp_pkmn.get('stats', {})
                    if stats:
                        context_parts.append(
                            f"  Battle Stats: ATK:{stats.get('attack', '?')} DEF:{stats.get('defense', '?')} SPD:{stats.get('speed', '?')}")

                    # Special indicators
                    if opp_pkmn.get('is_shiny'):
                        context_parts.append(f"  ✨ SHINY POKÉMON!")
                else:
                    # Opponent data not ready
                    context_parts.append(f"\n--- OPPONENT POKÉMON ---")
                    opponent_status = battle.get('opponent_status', 'Opponent data not available')
                    context_parts.append(f"⏳ {opponent_status}")
                    context_parts.append("  (Battle may be in initialization phase)")

            # Battle interface info
            interface = battle.get('battle_interface', {})
            available_actions = interface.get('available_actions', [])
            if available_actions:
                context_parts.append(f"\n--- AVAILABLE ACTIONS ---")
                context_parts.append(f"Options: {', '.join(available_actions)}")

            # Trainer battle specific info
            if battle.get('is_trainer_battle'):
                remaining = battle.get('opponent_team_remaining', 1)
                if remaining > 1:
                    context_parts.append(f"\nTrainer has {remaining} Pokémon remaining")

            # Battle phase info
            battle_phase = battle.get('battle_phase_name')
            if battle_phase:
                context_parts.append(f"\nBattle Phase: {battle_phase}")

        # Party information (important for switching decisions)
        context_parts.append("\n=== PARTY STATUS ===")
        party_context = _format_party_info(player_data, game_data)
        context_parts.extend(party_context)

        # Trainer info if available
        if 'name' in player_data and player_data['name']:
            context_parts.append(f"\nTrainer: {player_data['name']}")

        # Money/badges might be relevant
        money = player_data.get('money') or game_data.get('money')
        if money is not None:
            context_parts.append(f"Money: ${money}")

    else:
        # NORMAL MODE: Full state information
        context_parts.append("=== PLAYER INFO ===")

        # Player name and basic info
        if 'name' in player_data and player_data['name']:
            context_parts.append(f"Player Name: {player_data['name']}")

        # Position information
        position = _get_player_position(player_data)

        '''saved_map_position = world.last_obs_location if world else None

        print('DEBUG: player position:', position, 'saved_map position:', saved_map_position)'''

        if position:
            context_parts.append(f"Position: X={position.get('x', 'unknown')}, Y={position.get('y', 'unknown')}")

        # Facing direction
        # todo the memory read is bugged, get this from screenshot probably
        if 'facing' in player_data and player_data['facing']:
            context_parts.append(f"Facing: {player_data['facing']}")

        # Money (check both player and game sections)
        money = player_data.get('money') or game_data.get('money')
        if money is not None:
            context_parts.append(f"Money: ${money}")

        # Pokemon Party (check both player and game sections)
        party_context = _format_party_info(player_data, game_data)
        context_parts.extend(party_context)

        # Map/Location information with traversability (NOT shown in battle)
        map_info = state_data.get('map', {})
        if map_info: map_info['current_location'] = player_data.get('location')

        map_context, map_slice = _format_saved_map_info(map_info, world, include_debug_info, include_npcs)
        # map_context = _format_map_info(state_data.get('map', {}), include_debug_info, include_npcs)
        context_parts.extend(map_context)

        # Game state information (including dialogue if not in battle)
        game_context = _format_game_state(game_data)
        context_parts.extend(game_context)

    # Debug information if requested (shown in both modes)
    if include_debug_info:
        debug_context = _format_debug_info(state_data)
        context_parts.extend(debug_context)

    return "\n".join(context_parts), map_slice


def format_grid_for_llm(grid, show_coords=True):
    """Format a grid of symbols into a string for LLM input."""
    lines = []
    width = len(grid[0])
    num_places = None

    if show_coords:
        num_places = len(str(width))

        for place in reversed(range(num_places)):
            line = [' ' * (num_places + 2)]  # padding

            base = (10 ** place)
            num_values = width // base  # number of values in that place, e.g. 0-42 for tens: 4

            # n[-1] here because

            [line.extend([str(n)[-1]] * base) for n in range(0, num_values)]
            line += [str(num_values)[-1]] * (width % base)
            lines.append(" ".join(line))

    # header bar
    lines.append('-' * (len(lines[-1])))

    for ri, row in enumerate(grid):
        line = [f'{str(ri).rjust(num_places)} |'] if show_coords else []
        line += row
        lines.append(" ".join(line))

    map_display = "\n".join(lines)

    return map_display


# test
if __name__ == "__main__":
    import numpy as np

    grid = np.zeros((48, 152), dtype=int).astype('str')
    grid[0] = 'I'

    display = format_grid_for_llm(grid.tolist())

    legend = map_formatter.generate_dynamic_legend(grid.tolist())

    print(legend)


def _format_saved_map_info(map_info, world, include_debug_info=False, include_npcs=True, radius=20):
    """Format map and traversability information, combining world.mapper.Map
    and currently observed state data.
    """

    diameter = radius * 2 + 1

    context_parts = []

    context_parts.append("\n=== LOCATION & MAP INFO ===")

    context_parts.append(f"Current Map: TODO map name here")  # TODO get map name

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

    # Use raw tiles if available
    if 'tiles' in map_info and map_info['tiles']:
        raw_tiles = map_info['tiles']
        # Get player facing direction from context
        facing = "South"  # default

        # Get NPCs from map info
        npcs = map_info.get('object_events', []) if include_npcs else []
        # TODO figure out how this works, integrate into saved_map

        # Get player coordinates for NPC positioning
        player_coords = map_info.get('player_coords')

        # Use unified LLM formatter for consistency
        # map_display = format_map_for_llm(raw_tiles, facing, npcs, player_coords)



        # grid, offset = format_saved_map_grid(world, map_info.get('current_location'), facing, npcs, radius=radius)
        last_map = map_info.get('current_location')
        diameter = (radius * 2) + 1


        player_coords = world.last_global_location()

        if True: # TODO center_around_player
            center_y = player_coords[0]
            center_x = player_coords[1]
        else:
            raise NotImplementedError("Currently only center_around_player=True is supported")

        # Player is always at the center of the 15x15 grid view
        # but we need the actual player coordinates for NPC positioning
        # player_map_x = center_x  # Grid position (always 7,7 in 15x15)
        # player_map_y = center_y

        # Always use P for player instead of direction arrows
        player_symbol = "P"

        # Create NPC position lookup (convert to relative grid coordinates)
        npc_positions = {}
        if npcs and (player_coords is not None):
            try:
                # Handle both tuple and dict formats for player_coords
                if isinstance(player_coords, dict):
                    player_abs_x = player_coords.get('x', 0)
                    player_abs_y = player_coords.get('y', 0)
                else:
                    player_abs_x, player_abs_y = player_coords

                # Ensure coordinates are integers
                player_abs_x = int(player_abs_x) if player_abs_x is not None else 0
                player_abs_y = int(player_abs_y) if player_abs_y is not None else 0

                # TODO check the format of this
                for npc in npcs:
                    # NPCs have absolute world coordinates, convert to relative grid position
                    npc_abs_x = npc.get('current_x', 0)
                    npc_abs_y = npc.get('current_y', 0)

                    # Ensure NPC coordinates are integers
                    npc_abs_x = int(npc_abs_x) if npc_abs_x is not None else 0
                    npc_abs_y = int(npc_abs_y) if npc_abs_y is not None else 0

                    # Calculate offset from player in absolute coordinates
                    offset_x = npc_abs_x - player_abs_x
                    offset_y = npc_abs_y - player_abs_y

                    # Convert offset to grid position (player is at center)
                    grid_x = center_x + offset_x
                    grid_y = center_y + offset_y

                    # Check if NPC is within our grid view
                    if 0 <= grid_x < diameter and 0 <= grid_y < diameter:
                        npc_positions[(grid_y, grid_x)] = npc

            except (ValueError, TypeError) as e:
                # If coordinate conversion fails, skip NPC positioning
                print(f"Warning: Failed to convert coordinates for NPC positioning: {e}")
                print(f"  player_coords: {player_coords}")
                if npcs:
                    print(f"  npc coords: {[(npc.get('current_x'), npc.get('current_y')) for npc in npcs]}")
                npc_positions = {}

        # calculate slice of map to render

        # WHY IS THE BBOX EXPLODING?!

        map_slice = world.get_cropped_map(
            (center_y - radius, center_y + radius + 1,
             center_x - radius, center_x + radius + 1),
            last_map,
            crop_margin=1)

        grid = []

        for y, row in enumerate(map_slice.map):
            grid_row = []
            for x, tile in enumerate(row):
                map_y = y + map_slice.offset_y
                map_x = x + map_slice.offset_x

                if map_y == center_y and map_x == center_x:
                    # Player position
                    grid_row.append(player_symbol)
                elif (map_y, map_x) in npc_positions: # TODO this needs to be bundled in world
                    # NPC position - use NPC symbol
                    npc = npc_positions[(map_y, map_x)]
                    # Use different symbols for different NPC types
                    if npc.get('trainer_type', 0) > 0:
                        grid_row.append("@")  # Trainer
                    else:
                        grid_row.append("N")  # Regular NPC
                else:
                    # Regular tile
                    symbol = format_tile_to_symbol(tile)
                    grid_row.append(symbol)


            grid.append(grid_row)


        map_display = format_grid_for_llm(grid, show_coords=True)

        #

        context_parts.append(f"\n--- FULL TRAVERSABILITY MAP ({len(grid[0])}x{len(grid)}) ---")  # todo dynamic
        context_parts.append(map_display)

        # Add dynamic legend based on symbols in the map
        legend = map_formatter.generate_dynamic_legend(grid)
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

    return context_parts, map_slice


def format_saved_map_grid(world: World, last_map, player_facing="South", npcs=None, center_around_player=True, radius=20):
    """
    Format raw tile data into a traversability grid with NPCs.

    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Player facing direction for center marker
        npcs: List of NPC/object events with positions\
        radius: maximum radius to render for the LLM. This is in L1-norm (will produce a square grid of dimensions
        2 * radius + 1)

    Returns:
        list: 2D list of symbol strings
        tuple: (y_offset, x_offset) of the top-left corner of the grid in map coordinates. Adding this offset to the LLM output gets the saved_map coordinates.
    """

    diameter = (radius * 2) + 1

    grid = []
    # center_y = len(raw_tiles) // 2
    # center_x = len(raw_tiles[0]) // 2

    player_coords = world.last_global_location()

    if center_around_player:
        center_y = player_coords[0]
        center_x = player_coords[1]
    else:
        raise NotImplementedError("Currently only center_around_player=True is supported")

    # Player is always at the center of the 15x15 grid view
    # but we need the actual player coordinates for NPC positioning
    # player_map_x = center_x  # Grid position (always 7,7 in 15x15)
    # player_map_y = center_y

    # Always use P for player instead of direction arrows
    player_symbol = "P"

    # Create NPC position lookup (convert to relative grid coordinates)
    npc_positions = {}
    if npcs and (player_coords is not None):
        try:
            # Handle both tuple and dict formats for player_coords
            if isinstance(player_coords, dict):
                player_abs_x = player_coords.get('x', 0)
                player_abs_y = player_coords.get('y', 0)
            else:
                player_abs_x, player_abs_y = player_coords

            # Ensure coordinates are integers
            player_abs_x = int(player_abs_x) if player_abs_x is not None else 0
            player_abs_y = int(player_abs_y) if player_abs_y is not None else 0

            # TODO check the format of this
            for npc in npcs:
                # NPCs have absolute world coordinates, convert to relative grid position
                npc_abs_x = npc.get('current_x', 0)
                npc_abs_y = npc.get('current_y', 0)

                # Ensure NPC coordinates are integers
                npc_abs_x = int(npc_abs_x) if npc_abs_x is not None else 0
                npc_abs_y = int(npc_abs_y) if npc_abs_y is not None else 0

                # Calculate offset from player in absolute coordinates
                offset_x = npc_abs_x - player_abs_x
                offset_y = npc_abs_y - player_abs_y

                # Convert offset to grid position (player is at center)
                grid_x = center_x + offset_x
                grid_y = center_y + offset_y

                # Check if NPC is within our grid view
                if 0 <= grid_x < diameter and 0 <= grid_y < diameter:
                    npc_positions[(grid_y, grid_x)] = npc

        except (ValueError, TypeError) as e:
            # If coordinate conversion fails, skip NPC positioning
            print(f"Warning: Failed to convert coordinates for NPC positioning: {e}")
            print(f"  player_coords: {player_coords}")
            if npcs:
                print(f"  npc coords: {[(npc.get('current_x'), npc.get('current_y')) for npc in npcs]}")
            npc_positions = {}

    # calculate slice of map to render

    map_slice = world.get_cropped_map(
        (center_y - radius, center_y + radius + 1,
        center_x - radius, center_x + radius + 1),
        last_map,
        crop_margin=1)

    for y, row in enumerate(cropped_map):
        grid_row = []
        for x, tile in enumerate(row):
            map_y = y + y0
            map_x = x + x0

            if map_y == center_y and map_x == center_x:
                # Player position
                grid_row.append(player_symbol)
            elif (map_y, map_x) in npc_positions:
                # NPC position - use NPC symbol
                npc = npc_positions[(map_y, map_x)]
                # Use different symbols for different NPC types
                if npc.get('trainer_type', 0) > 0:
                    grid_row.append("@")  # Trainer
                else:
                    grid_row.append("N")  # Regular NPC
            else:
                # Regular tile
                symbol = format_tile_to_symbol(tile)
                grid_row.append(symbol)
        grid.append(grid_row)

    return grid, offset
