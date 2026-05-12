import os
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset


LABEL_MAPPING = {
    'bev_label': 'y_bev',
    'cog_label': 'y_cog',
    'god_label': 'y_god',
    'label_bev': 'y_bev',
    'label_cog': 'y_cog',
    'label_god': 'y_god',
    'label_path': 'labels_path',
    'src_label_path': 'labels_path',
    'labels_file': 'labels_path',
}


class Qwen3VLAuxPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        aux_labels = {}
        for source_key, target_key in LABEL_MAPPING.items():
            if source_key in row and target_key not in row:
                row[target_key] = row.pop(source_key)

        for key in ['y_bev', 'y_cog', 'y_god']:
            if key in row:
                aux_labels[key] = row.pop(key)
        if 'labels_path' in row:
            aux_labels['labels_path'] = row.pop('labels_path')

        row = super().preprocess(row)
        if row is None:
            return None

        row.update(aux_labels)
        return row


dataset_path = os.environ.get('QWEN3VL_AUX_DATASET_PATH')
if dataset_path:
    register_dataset(
        DatasetMeta(
            dataset_name='qwen3vl_aux_local',
            dataset_path=dataset_path,
            preprocess_func=Qwen3VLAuxPreprocessor(columns={
                'image': 'images',
                'video': 'videos',
                'bev_label': 'y_bev',
                'cog_label': 'y_cog',
                'god_label': 'y_god',
            }),
        ))


__all__ = ['Qwen3VLAuxPreprocessor']
