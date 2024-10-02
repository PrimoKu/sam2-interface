class ObjectManager:
    def __init__(self):
        self.objects = {}

    def add_object(self, obj_id, category_name, color):
        self.objects[obj_id] = {
            'category_name': category_name,
            'color': color,
            'last_valid_mask': None
        }

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

    def clear(self):
        self.objects.clear()