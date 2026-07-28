---
license: apache-2.0
library_name: PaddleOCR
language:
- en
- zh
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- textline_detection
---

# PP-OCRv4_mobile_det

## Introduction

PP-OCRv4_mobile_det is one of the PP-OCRv4_det series models, a set of text detection models developed by the PaddleOCR team. This mobile-optimized text detection model offers higher efficiency, making it ideal for deployment on edge devices. Its key accuracy metrics are as follows:

| Handwritten Chinese | Handwritten English | Printed Chinese | Printed English | Traditional Chinese | Ancient Text | Japanese | General Scenario | Pinyin | Rotation | Distortion | Artistic Text | Average | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.583 | 0.369 | 0.872 | 0.773	| 0.663 | 0.231 | 0.634	 | 0.710 | 0.430 | 0.299 | 0.715 | 0.549 | 0.624 |

## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleOCR

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```

### Model Usage

You can quickly experience the functionality with a single command:

```bash
paddleocr text_detection \
    --model_name PP-OCRv4_mobile_det \
    -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png
```

You can also integrate the model inference of the text detection module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv4_mobile_det")
output = model.predict(input="3ul2Rq4Sk5Cn-l69D695U.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/3ul2Rq4Sk5Cn-l69D695U.png', 'page_index': None, 'dt_polys': array([[[ 637, 1432],
        ...,
        [ 637, 1454]],

       ...,

       [[ 356,  107],
        ...,
        [ 356,  130]]], dtype=int16), 'dt_scores': [0.8305358711080322, 0.6912752452425651, ..., 0.848925772091929]}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/DnuBmbm8nDkuj_lLNS5fm.jpeg)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/text_detection.html#iii-quick-start).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-OCRv4

The general OCR pipeline is used to solve text recognition tasks by extracting text information from images and outputting it in text form. And there are 5 modules in the pipeline: 
* Document Image Orientation Classification Module (Optional)
* Text Image Unwarping Module (Optional)
* Text Line Orientation Classification Module (Optional)
* Text Detection Module
* Text Recognition Module

Run a single command to quickly experience the OCR pipeline:

```bash
paddleocr ocr -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png \
    --text_detection_model_name PP-OCRv4_mobile_det \
    --text_recognition_model_name PP-OCRv4_mobile_rec \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --save_path ./output \
    --device gpu:0 
```

Results are printed to the terminal:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/3ul2Rq4Sk5Cn-l69D695U.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[ 356,  105],
        ...,
        [ 356,  129]],

       ...,

       [[ 630, 1432],
        ...,
        [ 630, 1451]]], dtype=int16), 'text_det_params': {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['AlgorithmsfortheMarkovEntropyDecomposition', 'AndrewJ.FerrisandDavidPoulin', 'DepartementdePhysique,UniversitedeSherbrooke，Quebec,J1K2R1，Canada', '(Dated:October 31,2018)', 'TheMarkoventropydecomposition(MED)isarecently-proposed,cluster-basedsimulationmethodforfi-', 'nite temperature quantum systems with arbitrary geometry. In this paper, we detail numerical algorithms for', 'performingtherequiredsteps oftheMED,principallysolvingaminimizationproblemwithapreconditioned', '2107', "Newton's algorithm, as well as how to extract global susceptibilities and thermal responses. We demonstrate", 'thepowerof themethodwiththespin-1/2XXZmodelonthe2Dsquarelattice,includingtheextractionof', 'criticalpointsanddetailsofeachphase.Althoughthemethodsharessomequalitativesimilaritieswithexact-', 'diagonalization,we show the MED is both more accurate and significantly more fexible', '', 'PACS numbers: 05.10.a, 02.50.Ng, 03.67.a, 74.40.Kb', '6', '1', 'INTRODUCTION', 'This approximation becomes exactin the case of a1Dquan', 'tum (or classical)Markov chain[10],and leads to an expo', 'g', 'Althoughtheequationsgoverningquantummany-body', 'nentialreduction of costfor exact entropy calculationswhen', 'C', 'systemsares', 'simpletowritedown,findingsolutionsforthe', 'theglobaldensitymatrixisahigher-dimensionalMarkovnet-', 'H', 'majorityof systems remainsincrediblydifficult.Modern', 'work state[12,13].', 'physicsfinds itself inneedof new tools tocompute theemer-', 'Thesecond approximationused intheMEDapproach is', 'gent behavioroflarge,many-body systems.', 'relatedtotheN-representibilityproblem.Givenasetoflo', '', 'T', 'Therehasbeen a greatvariety of tools developed totackle', 'calbut overlappingreduceddensitymatrices{pi},itis avery', 'many-body problems,but in general, large 2D and 3D quan-', 'challengingproblemtodetermineifthereexistsaglobalden', '1', 'tumsystemsremainhardtodealwith.N', 'Mostsystemsare', 'sityoperatorwhichispositivesemi-definiteandwhosepartial', 'thoughttobenon-integrable,soexactanalyticsolutionsare', 'trace agreeswitheachpi.This problemis QMA-hard(the', 'notusuallyexpected.Directnumericaldiagonalizationcanbe', 'quantum analogue of NP)[14,15],and is hopelessly diffi', 'performedforrelativelysmallsystems', 'howevertheemer', 'cult toenforce.Thus,the second approximationemployed', 'gentbehaviorofasysteminthethermodynamiclimitmaybe', 'involves ignoringglobal consistency with apositive opera', 'difficulttoextract,especiallyins', 'systemswithlargecorrelation', 'tor,whilerequiringlocal consistency on any overlappingre', 'lengths.MonteCarloapproachesaretechnicallyexact(upto', 'gionsbetweenthep.Atthezero-temperaturelimit,theMED', 'samplingerror),butsufferfromtheso-calledsignproblem', 'approachbecomesanalogoustothevariationalnth-orderre-', 'forfermionic,frustrated,or dynamical problems.Thus we are', 'duceddensitymatrix', 'approach,wherepositivityisenforced', 'limited to search for clever approximations to solve the ma-', 'on allreduceddensitymatricesofsizen[16-18].', 'jorityofmany-bodyproblems', 'TheMEDapproachisanextremelyflexibleclustermethod', 'Over thepastcentury,hundredsof suchapproximations', 'applicabletobothtranslationallyinvariantsystemsofanydi', 'havebeenproposed,andwewillmentionjustafewnotable', 'mensioninthethermodynamiclimit,aswellasfinitesystems', 'examplesapplicabletoquantumlatticemodels.Mean-field', 'or systems without translationalinvariance(e.g.disordered', 'theoryiss', 'simplea', 'andfrequentlyarrivesatthecorrectquali', 'lattices,orharmonicallyt', 'trappeda', 'atomsinopticallattices)', 'tativedescription,butoftenfailswhencorrelationsareim', 'The free energy given by MED is guaranteed to lower bound', 'portant. Density-matrix renormalisation group (DMRG)[1]', 'the true free energy,which in turn lower-bounds the ground', 'is efficient and extremely accurate atsolving1Dproblems', 'stateenergy—t', 'thusprovidinganaturalcomplementtovaria', 'butthecomputationalcostgrowsexponentiallywithsystem', 'tional approacheswhichupper-bound thegroundstateenergy', 'sizeintwo-or higher-dimensions[2,3].F', 'Relatedtensor', 'Theabilitytoprovidearigorousground-stateenergywindow', 'networktechniquesdesignedfor2Dsystemsarestillinthein', 'is apowerfulvalidation tool,creating avery compellingrea-', 'infancy[4-6].Series-expansionmethods[7]canbesuccess-', 'son tousethis approach', 'ful,but may diverge or otherwise converge slowly,obscuring', 'Inthispaperwepaperwepresent apedagogicalintroduc', 'thestateincertainregimes.', 'Thereexistavarietyofcluster', 'tiontoMED,includingnumericalimplementationissuesand', 'basedtechniques,suchasdynamical-mean-fieldtheory[8]', 'applicationsto2Dquantumlatticemodelsinthethermody', 'anddensity-matrixembedding[9]', 'namiclimit.In Sec.I', 'II,wegiveabrief', 'derivationofthe', 'Herewe discuss theso-calledMarkoventropydecompo-', 'Markoventropydecomposition.SectionII outlines arobust', 'sition(MED),recentlyproposed byPoulin&Hastings [10]', 'numericalstrategyfor optimizingtheclusters thatmakeup', '(and analogoustoaslightlyearlier classical algorithm[11])', 'thedecomposition.InSec.IVweshowhowwecanextend', 'Thisisaself-consistentclustermethodforfinite temperature', 'thesealgorithmstoextractnon-trivialinformation,suchas', 'systems that takes advantage of an approximation of the(von', 'specificheat andsusceptibilities.Wepresentan application of', 'Neumann)entropy.In[1o],it was shown that the entropy', 'themethod to the spin-1/2XXZmodelon a 2Dsquarelattice', 'persitecanberigorouslyupperboundedusingonlylocalin-', 'inSec.V,describinghowtocharacterizethephasediagram', 'formation—alocal,reduced density matrix on Nsites,say.', '', 'and determine criticalpoints,before concluding inSec.VI.'], 'rec_scores': array([0.9952876 , ..., 0.95561302]), 'rec_polys': array([[[ 356,  105],
        ...,
        [ 356,  129]],

       ...,

       [[ 630, 1432],
        ...,
        [ 630, 1451]]], dtype=int16), 'rec_boxes': array([[ 356, ...,  130],
       ...,
       [ 630, ..., 1451]], dtype=int16)}}
```

If save_path is specified, the visualization results will be saved under `save_path`. The visualization output is shown below:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/g6n-H2VFG5ZYD0J8YnVfh.jpeg)

The command-line method is for quick experience. For project integration, also only a few codes are needed as well:

```python
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv4_mobile_det",
    text_recognition_model_name="PP-OCRv4_mobile_rec",
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
)
result = ocr.predict("./3ul2Rq4Sk5Cn-l69D695U.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")
```

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html#2-quick-start).


## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)
