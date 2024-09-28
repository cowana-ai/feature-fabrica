class TransformationRegistry:
    registry: dict[str, str] = {}

    @classmethod
    def register(cls, transformation_class):
        # Automatically infer the class path from the class
        class_path = f"{transformation_class.__module__}.{transformation_class.__name__}"
        if transformation_class._name_:
            cls.registry[transformation_class._name_] = class_path

    @classmethod
    def get_all_transformation_names(cls):
        return cls.registry

    @classmethod
    def get_transformation_class_by_name(cls, name: str):
        return cls.registry[name]
