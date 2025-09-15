# to map the world
import numpy as np
import pyastar2d


class World():
    # graph network connecting Maps
    # will do: spatial memory/context for each map

    def __init__(self, max_map_size=128, unknown_value=-1, channels=4):
        self.maps = []
        self.map_names = [] # each map's name as detected by the memory, or assigned by the agent
        self.max_map_size = max_map_size
        self.unknown_value = unknown_value
        self.channels = channels
        self.last_map_idx = None

    def observe(self, obs):
        if self.maps == []:
            self.maps += [Map(max_map_size=self.max_map_size, unknown_value=self.unknown_value, channels=self.channels)]
            self.last_map_idx = 0

        result = self.maps[self.last_map_idx].observe(obs)
        if result == 1: # check for warps TODO; just create a new map for now
            print('detected a warp (debug)')

            self.maps.append(Map(max_map_size=self.max_map_size, unknown_value=self.unknown_value))
            self.last_map_idx = len(self.maps) - 1
            
            self.maps[self.last_map_idx].observe(obs)

            '''for map in self.maps:
                map_result = map.full_observe(obs)
                continue'''

    def num_maps(self):
        return len(self.maps)


    def add_map(self):
        self.maps.append(Map(self.max_map_size, self.unknown_value))
        self.last_map_idx = len(self.maps) - 1

    def get_cropped_map(self, idx, channel_mask=None, get_coords=False):
        if self.maps == []:
            return None
        channel_mask = channel_mask or np.arange(self.channels)


        if get_coords:
            map, coords = self.maps[idx].get_cropped_map(get_coords=True)
            map = np.squeeze(map[:, :, channel_mask])
            return map, coords

        else:
            map = self.maps[idx].get_cropped_map()[:, :, channel_mask]
            return np.squeeze(map)

    def get_map(self, idx, channel_mask=None, get_coords=False):

        if self.maps == []:
            return None
        channel_mask = channel_mask or np.arange(self.channels)

        if get_coords:
            map, coords = self.maps[idx].get_map(get_coords=True)
            map = np.squeeze(map[:, :, channel_mask])
            return map, coords

        else:
            map = self.maps[idx].get_map()[:, :, channel_mask]
            return np.squeeze(map)

    def pathfind_on_current_map(self, goal, start=None, return_buttonpresses=False):
        if self.maps == []:
            return None

        return self.maps[self.last_map_idx].pathfind(goal, start, return_buttonpresses)


class Map(object):
    direction_mappings = { # WASD
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
        # Use a fixed square dense map for fast slicing/assignment
        self.max_map_size = int(max_map_size)
        self.unknown_value = unknown_value
        self.channels = channels
        self.map = np.full((self.max_map_size, self.max_map_size, self.channels), unknown_value, dtype=np.int32)


        # Center of the world in the fixed array (allows negative world coords)
        self.center = np.array([self.max_map_size // 2, self.max_map_size // 2], dtype=np.int32)

        self.last_obs = None
        self.last_obs_location = self.center
        self.warps = []

        # bounding box of known observations within self.map: [y_min, y_max_excl, x_min, x_max_excl]
        self.bbox = None

    def observe(self, obs, consistency_threshold=64):
        if isinstance(obs, list):
            obs = np.array(obs)

        if self.last_obs is None:
            self.add_to_map(obs, self.center)
            self.last_obs_location = self.center
            self.last_obs = obs
            # update bbox after write
            self._update_bbox_from_last_write(obs, self.center)

            return None

        # first, search for displacement from previous observation
        scores = []
        scores += [self.consistency_score(obs, self.last_obs)]
        scores += [self.consistency_score(obs[1:, :, :], self.last_obs[:-1, :, :])]
        scores += [self.consistency_score(obs[:, 1:, :], self.last_obs[:, :-1, :])]
        scores += [self.consistency_score(obs[:-1, :, :], self.last_obs[1:, :, :])]
        scores += [self.consistency_score(obs[:, :-1, :], self.last_obs[:, 1:, :])]


        direction = int(np.argmin(scores))

        if scores[direction] > consistency_threshold:
            # we probably warped
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

    def full_observe(self, obs):
        # observe, but compare against the entire map to find the current location
        # if nothing looks similar then return
        return

    def _update_bbox_from_last_write(self, obs, coords):
        # Compute the clipped destination slice we just wrote to, identical to add_to_map math
        h, w = obs.shape[:2]
        half_w = w // 2
        half_h = h // 2
        world_x0 = int(coords[0]) - half_w
        world_y0 = int(coords[1]) - half_h

        arr_y0 = world_y0 #  + int(self.center[1])
        arr_x0 = world_x0 #  + int(self.center[0])
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

    def get_cropped_map(self, get_coords=False):
        """
        Return the tight crop of the internal map using the current bounding box.
        If nothing has been observed yet, return an empty (0,0) array.
        """
        if self.bbox is None:
            return np.zeros((0, 0), dtype=self.map.dtype)
        y0, y1, x0, x1 = self.bbox

        if get_coords:
            clipped_coords = self.last_obs_location - np.array([x0, y0])
            return self.map[y0:y1, x0:x1, :], clipped_coords

        return self.map[y0:y1, x0:x1].copy()

    def get_map(self, get_coords=False):
        if get_coords:
            return self.map, self.last_obs_location
        return self.map.copy()


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
        arr_y0 = world_y0 # + int(self.center[1])
        arr_x0 = world_x0 # + int(self.center[0])

        # Compute in-bounds slice
        arr_y1 = arr_y0 + h
        arr_x1 = arr_x0 + w

        # Clip to map bounds
        clip_y0 = max(arr_y0, 0)
        clip_x0 = max(arr_x0, 0)
        clip_y1 = min(arr_y1, self.max_map_size)
        clip_x1 = min(arr_x1, self.max_map_size)

        if clip_y0 != arr_y0 or clip_x0 != arr_x0 or clip_y1 != arr_y1 or clip_x1 != arr_x1:
            print('observation is out of bounds!')

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
        weights[weights == -1] = 9999 # treat unknown tiles as impassable

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


