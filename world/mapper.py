# to map the world
import numpy as np
from collections import deque
import pyastar2d
from world.warps_reference import warps_reference


class World():
    # graph network connecting Maps
    # will do: spatial memory/context for each map

    def __init__(self, max_map_size=128, unknown_value=-1, channels=4):
        self.maps = {}

        self.map_bboxes = {}  # each map's bounding box in world coordinates
        self.map_strict_bboxes = {}  # each map's strict bounding box in world coordinates

        # self.map_names = [] # each map's name as detected by the memory, or assigned by the agent
        self.max_map_size = max_map_size
        self.unknown_value = unknown_value
        self.channels = channels
        self.last_map_name = None
        self.is_in_battle = False  # for most black-outs we warp back to pokecenter, this shouldn't be saved as a warp connection
        # TODO blacking out via poison is a rare way to warp in the overworld which would in this case be thought to be a warp connection

    def last_global_location(self):
        return self.maps[self.last_map_name].last_obs_location

    def current_map(self, return_coords=False):
        if self.maps == {}:
            return None
        if return_coords:
            return self.maps[self.last_map_name], self.maps[self.last_map_name].last_obs_location
        return self.maps[self.last_map_name]

    def observe(self, obs, memory_location=None, coords=None):

        if self.maps == {}:
            self.maps[memory_location] = Map(max_map_size=self.max_map_size, unknown_value=self.unknown_value,
                                             channels=self.channels)
            self.last_map_name = memory_location

        # try observing into the current map first

        result = self.maps[self.last_map_name].observe(obs, memory_location=memory_location, coords=coords)

        if result == 2:  # obs is connected to this map but is of a different memory_location - indicates a direct connection
            print('detected a new memory_location connected to current map (debug)')
            last_map = self.maps[self.last_map_name]

            if memory_location in self.maps.keys():
                new_map = self.maps[memory_location]


            else:
                # this has not been previously stored, so create a new map
                self.maps[memory_location] = Map(max_map_size=self.max_map_size, unknown_value=self.unknown_value)
                new_map = self.maps[memory_location]

            # new map should treat this like a normal observation

            new_result = new_map.observe(obs, memory_location=memory_location, coords=coords)
            assert new_result == 0

            # connect the two maps in the graph TODO how are coordinates across maps handled? last_obs_location may need additional handling/offset        

            new_map.connect(self.last_map_name, last_map.last_obs_location, new_map.last_obs_location, type='direct')
            last_map.connect(memory_location, new_map.last_obs_location, last_map.last_obs_location, type='direct')
            self.last_map_name = memory_location

        elif result == 1:  # obs is a different memory_location and not connected to the current map - indicates a warp

            if not self.is_in_battle:

                self.maps[memory_location] = Map(max_map_size=self.max_map_size, unknown_value=self.unknown_value)

                last_map = self.maps[self.last_map_name]
                new_map = self.maps[memory_location]

                new_map.connect(self.last_map_name, last_map.last_obs_location, new_map.last_obs_location, type='warp')
                last_map.connect(memory_location, new_map.last_obs_location, last_map.last_obs_location, type='warp')

                self.last_map_name = memory_location

                # new_map.observe(obs, memory_location=memory_location, coords=coords)
                # the coords don't actually change during the warp, so don't observe
            else:
                print('in battle, not creating a new map for warp')

                in_pokecenter = 'PokeCenter' in memory_location or 'BrendanHouse' in memory_location  # TODO find the correct token
                if in_pokecenter:
                    # this should be a previously visited pokecenter
                    if not memory_location in self.maps:
                        print(memory_location, 'not in maps so is being created, something went wrong!')
                        self.maps[memory_location] = Map(max_map_size=self.max_map_size,
                                                         unknown_value=self.unknown_value)

                    self.maps[memory_location].observe(obs, memory_location=memory_location, coords=coords)
                    self.last_map_name = memory_location
                else:
                    print('not in pokecenter, something went wrong!')

            '''
            for map in self.maps:
                map_result = map.full_observe(obs)
                continue
            '''

        elif result == 0:  # obs added to current map
            pass

        self.is_in_battle = False

    def num_maps(self):
        return len(self.maps.keys())

    '''def get_cropped_map(self, idx, channel_mask=None, get_coords=False): # TODO
        if self.maps == []:
            return None
        channel_mask = channel_mask or np.arange(self.channels)


        if get_coords:
            map, coords = self.maps[idx].get_cropped_map(get_coords=True)
            map = np.squeeze(map[:, :, channel_mask])
            return map, coords

        else:
            map = self.maps[idx].get_cropped_map()[:, :, channel_mask]
            return np.squeeze(map)'''

    def get_cropped_map(self, corners, seed_map, use_map_bboxes='none', channel_mask=None, crop_margin=0):
        """
            Returns a MapSlice object containing the cropped map constructed from this World's map objects that are
            directly connected to the 'seed' map.
            The slice is defined by the intersect of the provided corners and the specified bounding box.
            If use_map_bboxes is 'none', the provided corners are cropped to the map size TODO
        """
        y0, y1, x0, x1 = corners
        num_channel_dims = channel_mask.count_nonzero() if channel_mask is not None else self.channels
        map_slice = np.full((y1 - y0, x1 - x0, num_channel_dims), -1, dtype=np.int32)

        if not isinstance(seed_map, Map):
            seed_map = self.maps[seed_map]
        connected_maps = seed_map.connected_directs()
        connected_maps += [seed_map.memory_location]

        for map_name in connected_maps:
            submap = self.maps[map_name]
            # Convert world slice to seed map's local coordinates
            dy, dx = seed_map.get_connected_offset(map_name)

            local_x0 = x0 + dx
            local_x1 = x1 + dx
            local_y0 = y0 + dy
            local_y1 = y1 + dy

            # Calculate intersection in local coordinates
            slice, (sy0, sy1, sx0, sx1) = submap.get_cropped_map_slice(local_y0, local_y1, local_x0, local_x1,
                                                                       crop_margin=crop_margin, return_indices=True)

            if slice is not None:
                # convert back into world coordinates (cropped_...)
                cy0 = sy0 - dy
                cy1 = sy1 - dy
                cx0 = sx0 - dx
                cx1 = sx1 - dx

                map_slice[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = slice[
                    :, :, channel_mask] if channel_mask is not None else slice

        # remove -1

        mask = map_slice != -1

        # Check if there are any valid values at all
        if not np.any(mask):
            raise ValueError("Array contains only fill values (-1s). No bounding box can be found.")

        # Find the indices where the mask is True along each axis
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]

        # Get the min and max indices for rows and columns
        rmin, rmax = rows[0], rows[-1]
        cmin, cmax = cols[0], cols[-1]

        # Calculate the bounding box coordinates with margin
        # Use max() and min() to ensure we don't go beyond the array boundaries
        rmin_margin = max(0, rmin - crop_margin)
        rmax_margin = min(map_slice.shape[0], rmax + crop_margin + 1)  # +1 for slicing
        cmin_margin = max(0, cmin - crop_margin)
        cmax_margin = min(map_slice.shape[1], cmax + crop_margin + 1)  # +1 for slicing

        map_slice = map_slice[rmin_margin:rmax_margin, cmin_margin:cmax_margin]

        offset_x = x0 + cmin_margin # this is now the origin
        offset_y = y0 + rmin_margin # TODO add the map-to-map offset, currently this is
        # 0, 0 because we only generate from the map the player is on.

        return MapSlice(map_slice, offset_x=offset_x, offset_y=offset_y,
                        last_obs_location=self.current_map().last_obs_location - np.array([offset_y, offset_x]))

    def get_map(self, name, channel_mask=None, get_coords=False):

        if self.maps == []:
            return None

        channel_mask = channel_mask or np.arange(self.channels)

        if get_coords:
            map, coords = self.maps[name].get_map(get_coords=True)
            map = np.squeeze(map[:, :, channel_mask])
            return map, coords

        else:
            map = self.maps[name].get_map()[:, :, channel_mask]
            return np.squeeze(map)

    def pathfind_on_current_map(self, goal, start=None, return_buttonpresses=False):  # TODO
        if self.maps == []:
            return None

        return self.maps[self.last_map_name].pathfind(goal, start, return_buttonpresses)

    # TODO graph retrieval methods
    # TODO area region grid method


class Map(object):
    direction_mappings = {  # WASD
        0: np.array([0, 0]),
        1: np.array([-1, 0]),
        2: np.array([0, -1]),
        3: np.array([1, 0]),
        4: np.array([0, 1])
    }

    direction_names = {
        0: 'none',
        1: 'W',
        2: 'A',
        3: 'S',
        4: 'D'
    }

    def __init__(self, max_map_size=1024, unknown_value=-1, channels=4):

        self.memory_location = None  # the memory location this map corresponds to

        self.max_map_size = int(max_map_size)
        self.unknown_value = unknown_value
        self.channels = channels
        self.map = np.full((self.max_map_size, self.max_map_size, self.channels), unknown_value, dtype=np.int32)

        # Center of the world in storage coordinates
        # Currently, there is no internal shifting of the map

        self.center = np.array([self.max_map_size // 2, self.max_map_size // 2], dtype=np.int32)

        self.last_obs = deque(maxlen=5)  # queue of length 5
        self.last_memory_location = deque(maxlen=5)

        self.last_obs_location = self.center

        self.warps = {}
        self.directs = {}

        self.bbox = None  # bounding box of known observations within self.map: [y_min, y_max_excl, x_min, x_max_excl]
        self.strict_bbox = None  # only encompasses tiles known to be of this memory_location

    def _check_cardinal_directions(self, obs1, obs2, consistency_threshold=64):
        scores = []
        scores += [self.consistency_score(obs1, obs2)]
        scores += [self.consistency_score(obs1[1:, :, :], obs2[:-1, :, :])]
        scores += [self.consistency_score(obs1[:, 1:, :], obs2[:, :-1, :])]
        scores += [self.consistency_score(obs1[:-1, :, :], obs2[1:, :, :])]
        scores += [self.consistency_score(obs1[:, :-1, :], obs2[:, 1:, :])]

        direction = int(np.argmin(scores))

        if scores[direction] > consistency_threshold:
            return None
        else:
            return direction

    def observe(self, obs, memory_location, coords, consistency_threshold=64):
        # NOTE: COORDS FORMAT IS (X, Y)

        return_code = 0

        if isinstance(obs, list):
            obs = np.array(obs)

        if self.memory_location is None:
            self.memory_location = memory_location

        if not memory_location == self.memory_location:
            # check using adjacencies if we are close to any previous areas
            if len(self.last_obs) > 0:  # only check if we have previous observations
                direction = self._check_cardinal_directions(obs, self.last_obs[0],
                                                            consistency_threshold=consistency_threshold)
            else:
                direction = None

            # for now, use hardcoded reference to do warp detection for us
            if (ref := warps_reference.get((self.memory_location, memory_location))) is not None:
                if ref == 'warp':
                    print('detected new memory_location - warp (from reference)')
                    return_code = 1
                if ref == 'direct':
                    print('detected new memory_location - no warp (from reference)')
                    return_code = 2

            else:

                # traditional detection (broken)
                if direction is None:
                    # we warped
                    print('detected new memory_location with no adjacent directions - warp')
                    return_code = 1
                else:
                    # new memory location which is connected to the previous one

                    print('detected new memory_location with direction - no warp')
                    print(Map.direction_names[direction])

                    coords = self.last_obs_location + Map.direction_mappings[int(direction)]

                    self.add_to_map(obs, coords)

                    self._update_bbox_from_last_write(obs, coords, update_strict=False)

                    # this will be connected by the World class
                    return_code = 2

        else:  # is in current map

            self.add_to_map(obs, coords)  # TODO do we need to handle negative coordinates?
            self._update_bbox_from_last_write(obs, coords)

            return_code = 0

        self.last_memory_location.appendleft(memory_location)
        self.last_obs_location = coords
        self.last_obs.appendleft(obs)  # add to front of queue

        return return_code

        '''
        # this is now deprecated as long as we have coords
        # else handle by checking displacements from last observation
        # first, search for displacement from previous observation
        direction = self._check_cardinal_directions(obs, self.last_obs, consistency_threshold=consistency_threshold)
        if direction is None:
            return 1
        else:
            if direction != 0: # we stayed in place and didn't observe anything new
                # TODO except that moving NPCs can technically be a new observation

                print(scores)
                print(Map.direction_names[direction])

                coords = self.last_obs_location + Map.direction_mappings[int(direction)]
                self.add_to_map(obs, coords)
                self.last_obs_location = coords
                self.last_obs = obs
                # update bbox after write
                self._update_bbox_from_last_write(obs, coords)

        return 0
'''

    def full_observe(self, obs):
        # observe, but compare against the entire map to find the current location
        # if nothing looks similar then return
        return

    def _update_bbox_from_last_write(self, obs, coords, update_strict=True):
        # Compute the clipped destination slice we just wrote to, identical to add_to_map math
        # coords are (x, y) # TODO
        # Note this is python slice notation, so y1 and x1 are exclusive

        h, w = obs.shape[:2]
        half_w = w // 2
        half_h = h // 2
        world_y0 = int(coords[0]) - half_w
        world_x0 = int(coords[1]) - half_h

        arr_y0 = world_y0  # + int(self.center[1])
        arr_x0 = world_x0  # + int(self.center[0])
        arr_y1 = arr_y0 + h
        arr_x1 = arr_x0 + w

        clip_y0 = max(arr_y0, 0)
        clip_x0 = max(arr_x0, 0)
        clip_y1 = min(arr_y1, self.max_map_size)
        clip_x1 = min(arr_x1, self.max_map_size)

        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            return  # nothing written

        if self.bbox is None:
            self.bbox = [clip_y0, clip_y1, clip_x0, clip_x1]
        else:
            y0, y1, x0, x1 = self.bbox
            self.bbox = [
                min(y0, clip_y0),
                max(y1, clip_y1),
                min(x0, clip_x0),
                max(x1, clip_x1),
            ]

        if update_strict:
            if self.strict_bbox is None:
                self.strict_bbox = [coords[1], coords[1] + 1, coords[0], coords[0] + 1]
            else:
                y0, y1, x0, x1 = self.strict_bbox

                y0 = min(y0, coords[0])
                y1 = max(y1, coords[0] + 1)
                x0 = max(x0, coords[1])
                x1 = max(x1, coords[1] + 1)

                self.strict_bbox = [y0, y1, x0, x1]

        return

    def get_cropped_map(self, get_coords=False, use_strict_bbox=False):
        """
        Return the tight crop of the internal map using the current bounding box.
        If nothing has been observed yet, return an empty (0,0) array.
        """
        bbox = self.strict_bbox if use_strict_bbox else self.bbox
        if bbox is None:
            return np.zeros((0, 0), dtype=self.map.dtype)
        y0, y1, x0, x1 = bbox

        if get_coords:
            clipped_coords = self.last_obs_location - np.array([x0, y0])
            return self.map[y0:y1, x0:x1, :], clipped_coords

        return self.map[y0:y1, x0:x1].copy()

    def get_map(self, get_coords=False):
        if get_coords:
            return self.map, self.last_obs_location
        return self.map.copy()

    def get_cropped_map_slice(self, y0, y1, x0, x1, crop_margin=0, use_strict_bbox=False, return_indices=False):
        """
        Returns cropped map defined by the intersect of the provided slice in local coordinates and the current bounding box, with optional margin.
        If nothing has been observed yet, return an empty (0,0) array.
        If return_indices is True, also return the (y0, y1, x0, x1) indices of the returned crop within the full map.
        """

        bbox = self.strict_bbox if use_strict_bbox else self.bbox

        y0 = max(y0, bbox[0] - crop_margin, 0)
        y1 = min(y1, bbox[1] + crop_margin, self.max_map_size)
        x0 = max(x0, bbox[2] - crop_margin, 0)
        x1 = min(x1, bbox[3] + crop_margin, self.max_map_size)

        # is the intersection valid?
        if x0 >= x1 or y0 >= y1:
            return None

        if return_indices:
            return self.map[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)

        return self.map[y0:y1, x0:x1].copy()

    def add_to_map(self, obs, coords):
        # obs: HxWxC array; coords: np.array([x, y]) center of the observation in the world

        if obs is None:
            return

        h, w = obs.shape[:2]

        # top-left world coords (x,y)
        half_w = w // 2
        half_h = h // 2
        world_y0 = int(coords[0]) - half_w
        world_x0 = int(coords[1]) - half_h

        # map (array) indices: row = y, col = x, shifted by self.center
        arr_y0 = world_y0  # + int(self.center[1])
        arr_x0 = world_x0  # + int(self.center[0])

        # Compute in-bounds slice
        arr_y1 = arr_y0 + h
        arr_x1 = arr_x0 + w

        # Clip to map bounds
        clip_y0 = max(arr_y0, 0)
        clip_x0 = max(arr_x0, 0)
        clip_y1 = min(arr_y1, self.max_map_size)
        clip_x1 = min(arr_x1, self.max_map_size)

        # if clip_y0 != arr_y0 or clip_x0 != arr_x0 or clip_y1 != arr_y1 or clip_x1 != arr_x1:
        #    print('observation is out of bounds!')
        # currently we don't append out of bounds obs TODO shift the internal map representation

        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            # Entire observation is out of bounds; ignore
            return

        # Corresponding slice in obs
        src_y0 = clip_y0 - arr_y0
        src_x0 = clip_x0 - arr_x0
        src_y1 = src_y0 + (clip_y1 - clip_y0)
        src_x1 = src_x0 + (clip_x1 - clip_x0)

        # Assign
        self.map[clip_y0:clip_y1, clip_x0:clip_x1] = obs[src_y0:src_y1, src_x0:src_x1]

    def consistency_score(self, obs1, obs2):
        return np.sum(obs1 != obs2)

    def pathfind(self, goal, start=None, return_buttonpresses=False):
        # simple a-star pathfinding
        # TODO make a-star prioritize straight lines when possible (easier to handle)
        # TODO d star lite pathfinding?

        if start is None:
            start = self.last_obs_location

        # channel 2 is the collision channel
        weights = self.map[:, :, 2].astype(np.float32)

        weights[weights == 1] = 9999
        weights[weights == 0] = 1.0
        weights[weights == -1] = 9999  # treat unknown tiles as impassable

        # TODO: handle specially behaving tiles like warps/grass
        path = pyastar2d.astar_path(weights, start, goal)

        if path is None:
            return None

        # convert into button presses
        deltas = path[1:, :] - path[:-1, :]

        if return_buttonpresses:
            return 'bad'
        else:
            return deltas

    def connected_directs(self):
        return [mloc for mloc, *_ in self.directs.values()]

    def connect(self, other_name, other_tile, self_tile, type):
        if type == 'warp':  # one-way handling (??)
            self.warps[tuple(self_tile)] = [other_name, other_tile]

        elif type == 'direct':
            self.directs[tuple(self_tile)] = [other_name, other_tile]

    def get_connected_offset(self, other_name):
        # returns offset from self coordinate system to map to other's coordinate system.
        # essentially, self_coords + offset = other_coords

        if other_name == self.memory_location:
            return (0, 0)

        map_locations = [loc for loc, _ in list(self.directs.values())]

        if not other_name in map_locations:
            return None

        map_idx = map_locations.index(other_name)
        if map_idx != -1:
            self_tile = list(self.directs.keys())[map_idx]
            other_tile = self.directs[self_tile][1]

            return other_tile[0] - self_tile[0], other_tile[1] - self_tile[1]

        else:
            return None


class MapSlice():
    def __init__(self, map, offset_x=0, offset_y=0, last_obs_location=None):
        self.map = map
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.last_obs_location = last_obs_location

    def pathfind(self, goal, start=None, return_buttonpresses=False, reduce=True):
        # simple a-star pathfinding in local coordinates
        # TODO make a-star prioritize straight lines when possible (easier to handle)
        # TODO d star lite pathfinding?

        if start is None:
            start = self.last_obs_location

        # channel 2 is the collision channel
        weights = self.map[:, :, 2].astype(np.float32)

        weights[weights == 1] = 9999
        weights[weights == 0] = 1.0
        weights[weights == -1] = 9999  # treat unknown tiles as impassable

        if reduce:
            goal = max(0, min(goal[0], self.map.shape[0] - 1)), max(0, min(goal[1], self.map.shape[1] - 1))

        # TODO: handle specially behaving tiles like warps/grass
        path = pyastar2d.astar_path(weights, start, goal)

        if path is None:
            return None

        # convert into button presses
        deltas = path[1:, :] - path[:-1, :]

        if return_buttonpresses:
            return 'bad'
        else:
            return deltas
