import h5py
import json
import shutil

input_model = "models/fruit_classifier/best_mobilenetv2_apple_orange.h5"

output_model = "models/fruit_classifier/fixed_mobilenetv2_apple_orange.h5"


# Copy model asli agar tidak rusak
shutil.copy(input_model, output_model)


with h5py.File(output_model, "r+") as f:

    model_config = f.attrs.get("model_config")

    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    config = json.loads(model_config)


    def remove_quantization_config(obj):

        if isinstance(obj, dict):

            if "quantization_config" in obj:
                del obj["quantization_config"]

            for value in obj.values():
                remove_quantization_config(value)

        elif isinstance(obj, list):

            for item in obj:
                remove_quantization_config(item)


    remove_quantization_config(config)


    # Hapus model_config lama
    del f.attrs["model_config"]


    # Simpan config baru
    f.attrs["model_config"] = json.dumps(config)


print("Model berhasil diperbaiki!")
print(f"Saved as: {output_model}")