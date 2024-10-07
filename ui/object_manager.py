class ObjectManager:
    def __init__(self):
        self.objects = {}
        self.tracked_objects = set()
        self.non_tracked_objects = set()

    def add_object(self, obj_id, category_name, color, tracking=True):
        self.objects[obj_id] = {
            'category_name': category_name,
            'color': color,
            'last_valid_mask': None,
            'tracking': tracking
        }
        self._update_tracking_sets(obj_id, tracking)

    def set_tracking(self, obj_id, tracking):
        if obj_id in self.objects:
            self.objects[obj_id]['tracking'] = tracking
            self._update_tracking_sets(obj_id, tracking)

    def _update_tracking_sets(self, obj_id, tracking):
        if tracking:
            self.tracked_objects.add(obj_id)
            self.non_tracked_objects.discard(obj_id)
        else:
            self.non_tracked_objects.add(obj_id)
            self.tracked_objects.discard(obj_id)

    def get_tracked_objects(self):
        return list(self.tracked_objects)

    def get_non_tracked_objects(self):
        return list(self.non_tracked_objects)

    def get_object(self, obj_id):
        return self.objects.get(obj_id)

    def update_last_valid_mask(self, obj_id, mask):
        if obj_id in self.objects:
            self.objects[obj_id]['last_valid_mask'] = mask

    def get_all_objects(self):
        return self.objects

    def update_category_name(self, obj_id, new_name):
        if obj_id in self.objects:
            self.objects[obj_id]['category_name'] = new_name

    def remove_object(self, obj_id):
        if obj_id in self.objects:
            del self.objects[obj_id]
            self.tracked_objects.discard(obj_id)
            self.non_tracked_objects.discard(obj_id)

    def clear(self):
        self.objects.clear()
        self.tracked_objects.clear()
        self.non_tracked_objects.clear()