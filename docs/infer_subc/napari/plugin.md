# organelle-segmenter-plugin

The `infer_subc` Segementation Workflows are available through [Napari](https://napari.org/stable/) via the [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin) which was co-developed with 'infer_subc'

`organelle-segementer-plugin` leverages the logic created in the `Allen Cell & Structure Segmenter` [`napari plugin`](https://www.napari-hub.org/plugins/napari-allencell-segmenter). Although the logic of our **multi-channel** organelle segmentations required us to fork and modify their code, we hope it porvides a stable, but evolving base which will help manage accumulation of technical debt. In addition to the overall logic, we particulary leverage their *workflow* paradigm which is integral in the use of the napari plugin interface. Implementation of `infer-subc` as a Napari plugin using this framework is called [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin).

