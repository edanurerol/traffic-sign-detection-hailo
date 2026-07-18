from pathlib import Path
import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    HailoStreamInterface,
)


class HailoYoloInference:
    """
    HailoRT 4.23 uyumlu inference sınıfı.

    Görevi:
    - HEF dosyasını yüklemek
    - Hailo cihazını configure etmek
    - Network group'u aktif etmek
    - Tek frame/batch inference yapmak

    Not:
    Video okuma, NMS ve görselleştirme bu dosyada yapılmaz.
    """

    def __init__(self, hef_path: Path):
        self.hef_path = Path(hef_path)

        if not self.hef_path.exists():
            raise FileNotFoundError(f"HEF dosyası bulunamadı: {self.hef_path}")

        self.hef = HEF(str(self.hef_path))

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()

        self.input_name = self.input_info.name
        self.output_names = [info.name for info in self.output_infos]

        self.vdevice = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.activation_context = None
        self.infer_pipeline = None

    def __enter__(self):
        print("Hailo VDevice açılıyor...")
        self.vdevice = VDevice()

        print("HEF configure parametreleri oluşturuluyor...")
        configure_params = ConfigureParams.create_from_hef(
            self.hef,
            interface=HailoStreamInterface.PCIe,
        )

        print("Network group configure ediliyor...")
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        print("Input/Output vstream parametreleri oluşturuluyor...")
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32,
        )

        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32,
        )

        # HailoRT 4.23 için önemli sıra:
        # 1) network_group.activate(...)
        # 2) InferVStreams(...)
        print("Network group aktive ediliyor...")
        self.activation_context = self.network_group.activate(self.network_group_params)
        self.activation_context.__enter__()

        print("InferVStreams açılıyor...")
        self.infer_pipeline = InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        )
        self.infer_pipeline.__enter__()

        print("Hailo inference pipeline hazır.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Hailo kaynakları kapatılıyor...")

        if self.infer_pipeline is not None:
            try:
                self.infer_pipeline.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print("InferVStreams kapatılırken uyarı:", e)

        if self.activation_context is not None:
            try:
                self.activation_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print("Activation context kapatılırken uyarı:", e)

        if self.vdevice is not None:
            try:
                self.vdevice.release()
            except Exception:
                pass

    def infer(self, input_image: np.ndarray):
        """
        input_image:
        - shape: (640, 640, 3)
        - dtype: uint8 veya float32
        """

        if input_image.ndim != 3:
            raise ValueError(f"Beklenen input shape (H,W,C), gelen: {input_image.shape}")
        if input_image.shape != (640, 640, 3):
            raise ValueError(
                f"Beklenen input shape (640,640,3), gelen: {input_image.shape}"
            )

        input_batch = np.expand_dims(input_image.astype(np.float32), axis=0)

        input_data = {
            self.input_name: input_batch
        }

        results = self.infer_pipeline.infer(input_data)
        return results
