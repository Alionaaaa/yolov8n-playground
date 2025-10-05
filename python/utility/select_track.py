class SelectedTrack:
    def __init__(self):
        self.trk = None     # track data [x1, y1, x2, y2, id, cls, conf]
        self.index = -1     # index in tracks list
        self.valid = False  # whether there is a selected track

    def set_from_tracks(self, tracks, index=0):
        """Select a track by index from the tracks list"""
        if len(tracks) == 0:
            self.trk = None
            self.index = -1
            self.valid = False
            return
        self.index = index % len(tracks)
        self.trk = tracks[self.index]
        self.valid = True

    
    def move(self, tracks, step):
        if len(tracks) == 0:
            self.trk = None
            self.index = -1
            self.valid = False
            return

        # Get list of all IDs of current tracks
        track_ids = [int(t[4]) for t in tracks]

        if not self.valid:
            # if no track is selected — take the first one (with minimal ID)
            min_id = min(track_ids)
            self.index = track_ids.index(min_id)
            self.trk = tracks[self.index]
            self.valid = True
            return

        current_id = int(self.trk[4])
        if current_id not in track_ids:
            # if current track disappeared — select track with minimal ID
            min_id = min(track_ids)
            self.index = track_ids.index(min_id)
            self.trk = tracks[self.index]
            self.valid = True
            return

        # Shift ID in circular manner
        current_pos = track_ids.index(current_id)
        new_pos = (current_pos + step) % len(track_ids)
        self.index = new_pos
        self.trk = tracks[self.index]
        self.valid = True

    def update_if_needed(self, tracks):
        """Updates reference to track if it disappeared or order changed"""
        if not self.valid or len(tracks) == 0:
            self.set_from_tracks(tracks, 0)
            return
        track_ids = [int(t[4]) for t in tracks]
        current_id = int(self.trk[4])
        if current_id in track_ids:
            self.index = track_ids.index(current_id)
            self.trk = tracks[self.index]
        else:
            # if current track is missing — select by default the smallest ID
            min_id = min(track_ids)
            self.index = track_ids.index(min_id)
            self.trk = tracks[self.index]