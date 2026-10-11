"""
DERMATEC — Funciones de segmentación rápida y de los 10 parámetros dermatoscópicos.
Copia literal de las funciones de DERMATEC_J2_1_VISUAL/src/dermatec_j21_inferencia.py
(crop_black_border, resize_for_seg, segment_lesion, extract_features, make_roi, ...),
SIN la clase DermatecJ21 ni la importación de TensorFlow: no carga ningún modelo.
Así la app final no depende del paquete J2.1 ni de TensorFlow para los parámetros.
Rangos de referencia (p05/p50/p95 del conjunto Train): parametros_referencia.json
"""

from pathlib import Path

import json
import cv2
import numpy as np


CLASS_NAMES = [
    "Nevo",
    "Melanoma",
    "Carcinoma basocelular",
    "Carcinoma escamocelular"
]


# =================================================================================================
# UTILIDADES
# =================================================================================================

def largest_component(binary):

    binary = (
        binary > 0
    ).astype(
        np.uint8
    )

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if n <= 1:

        return binary

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    winner = (
        1
        +
        int(
            np.argmax(
                areas
            )
        )
    )

    return (
        labels == winner
    ).astype(
        np.uint8
    )


def odd_kernel(
    value,
    minimum=3,
    maximum=25
):

    k = int(
        round(
            value
        )
    )

    k = max(
        minimum,
        min(
            maximum,
            k
        )
    )

    if k % 2 == 0:

        k += 1

    return k


def crop_black_border(rgb):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    valid = (
        gray > 8
    ).astype(
        np.uint8
    )

    valid = cv2.morphologyEx(
        valid,
        cv2.MORPH_CLOSE,
        np.ones(
            (
                9,
                9
            ),
            dtype=np.uint8
        )
    )

    valid = largest_component(
        valid
    )

    ys, xs = np.where(
        valid > 0
    )

    if len(
        xs
    ) < 100:

        return rgb

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    h, w = rgb.shape[
        :2
    ]

    crop_fraction = (
        (
            x2 - x1 + 1
        )
        *
        (
            y2 - y1 + 1
        )
        /
        float(
            h * w
        )
    )

    if crop_fraction < 0.35:

        return rgb

    return rgb[
        y1:y2 + 1,
        x1:x2 + 1
    ]


def resize_for_seg(
    rgb,
    max_side=384
):

    h, w = rgb.shape[
        :2
    ]

    largest = max(
        h,
        w
    )

    if largest <= max_side:

        return rgb.copy()

    scale = (
        max_side
        /
        largest
    )

    nw = int(
        round(
            w * scale
        )
    )

    nh = int(
        round(
            h * scale
        )
    )

    return cv2.resize(
        rgb,
        (
            nw,
            nh
        ),
        interpolation=cv2.INTER_AREA
    )


# =================================================================================================
# SEGMENTACIÓN J2.1
# =================================================================================================

def component_score(
    mask,
    lab
):

    h, w = mask.shape

    area_px = float(
        mask.sum()
    )

    fraction = (
        area_px
        /
        float(
            h * w
        )
    )

    if (
        fraction < 0.015
        or
        fraction > 0.78
    ):

        return None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = float(
        cv2.contourArea(
            contour
        )
    )

    if area <= 0:

        return None

    hull = cv2.convexHull(
        contour
    )

    hull_area = max(
        float(
            cv2.contourArea(
                hull
            )
        ),
        1.0
    )

    solidity = (
        area
        /
        hull_area
    )

    moments = cv2.moments(
        contour
    )

    if moments[
        "m00"
    ] <= 0:

        return None

    cx = (
        moments[
            "m10"
        ]
        /
        moments[
            "m00"
        ]
    )

    cy = (
        moments[
            "m01"
        ]
        /
        moments[
            "m00"
        ]
    )

    distance = (
        np.sqrt(
            (
                cx - w / 2
            ) ** 2
            +
            (
                cy - h / 2
            ) ** 2
        )
        /
        max(
            np.sqrt(
                (
                    w / 2
                ) ** 2
                +
                (
                    h / 2
                ) ** 2
            ),
            1.0
        )
    )

    x, y, bw, bh = cv2.boundingRect(
        contour
    )

    touches = float(
        x <= 2
        or
        y <= 2
        or
        x + bw >= w - 2
        or
        y + bh >= h - 2
    )

    ring_k = odd_kernel(
        min(
            h,
            w
        )
        *
        0.05,
        7,
        21
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            ring_k,
            ring_k
        )
    )

    dilated = cv2.dilate(
        mask,
        kernel
    )

    ring = (
        (dilated > 0)
        &
        (mask == 0)
    )

    lesion_pixels = lab[
        mask > 0
    ]

    skin_pixels = lab[
        ring
    ]

    contrast = 0.0

    if (
        len(
            lesion_pixels
        ) > 50
        and
        len(
            skin_pixels
        ) > 50
    ):

        delta = (
            lesion_pixels.mean(
                axis=0
            )
            -
            skin_pixels.mean(
                axis=0
            )
        )

        contrast = float(
            np.clip(
                np.linalg.norm(
                    delta
                )
                /
                70.0,
                0.0,
                1.0
            )
        )

    area_preference = float(
        np.exp(
            -(
                (
                    fraction - 0.28
                )
                /
                0.30
            ) ** 2
        )
    )

    score = (
        1.25
        *
        (
            1.0
            -
            np.clip(
                distance,
                0,
                1
            )
        )
        +
        0.65
        *
        area_preference
        +
        0.45
        *
        solidity
        +
        0.55
        *
        contrast
        -
        0.50
        *
        touches
    )

    return {
        "score":
            float(
                score
            ),

        "area_fraction":
            float(
                fraction
            ),

        "solidity":
            float(
                solidity
            )
    }


def segment_lesion(
    rgb
):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    h, w = gray.shape

    channels = [
        gray,
        hsv[:, :, 1],
        hsv[:, :, 2],
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2]
    ]

    open_k = odd_kernel(
        min(
            h,
            w
        )
        /
        120,
        3,
        7
    )

    close_k = odd_kernel(
        min(
            h,
            w
        )
        /
        45,
        5,
        17
    )

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            open_k,
            open_k
        )
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            close_k,
            close_k
        )
    )

    best = None

    for channel in channels:

        blurred = cv2.GaussianBlur(
            channel,
            (
                5,
                5
            ),
            0
        )

        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY
            +
            cv2.THRESH_OTSU
        )

        for polarity in [
            0,
            1
        ]:

            if polarity == 0:

                binary = (
                    otsu > 0
                ).astype(
                    np.uint8
                )

            else:

                binary = (
                    otsu == 0
                ).astype(
                    np.uint8
                )

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel_close
            )

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel_open
            )

            n, labels, _, _ = (
                cv2.connectedComponentsWithStats(
                    binary,
                    connectivity=8
                )
            )

            for component_id in range(
                1,
                n
            ):

                component = (
                    labels
                    ==
                    component_id
                ).astype(
                    np.uint8
                )

                evaluation = component_score(
                    component,
                    lab
                )

                if evaluation is None:

                    continue

                if (
                    best is None
                    or
                    evaluation[
                        "score"
                    ]
                    >
                    best[
                        "score"
                    ]
                ):

                    contours, _ = cv2.findContours(
                        component,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    contour = max(
                        contours,
                        key=cv2.contourArea
                    )

                    filled = np.zeros_like(
                        component
                    )

                    cv2.drawContours(
                        filled,
                        [
                            contour
                        ],
                        -1,
                        1,
                        thickness=-1
                    )

                    best = {
                        **evaluation,
                        "mask":
                            filled
                    }

    if best is None:

        return {
            "mask":
                np.zeros(
                    (
                        h,
                        w
                    ),
                    dtype=np.uint8
                ),

            "valid":
                0,

            "quality":
                0.0
        }

    quality = float(
        np.clip(
            (
                best[
                    "score"
                ]
                -
                0.75
            )
            /
            2.20,
            0.0,
            1.0
        )
    )

    valid = int(
        quality >= 0.28
        and
        best[
            "area_fraction"
        ] >= 0.02
        and
        best[
            "area_fraction"
        ] <= 0.75
    )

    return {
        **best,

        "valid":
            valid,

        "quality":
            quality
    }


# =================================================================================================
# RADIOMICS J2.1
# =================================================================================================

FEATURE_COLUMNS = [

    "area_relative",
    "perimeter_normalized",
    "circularity",
    "irregularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "eccentricity",

    "asymmetry_horizontal",
    "asymmetry_vertical",
    "asymmetry_mean",

    "rgb_r_mean",
    "rgb_g_mean",
    "rgb_b_mean",

    "rgb_r_std",
    "rgb_g_std",
    "rgb_b_std",

    "hsv_s_mean",
    "hsv_s_std",
    "hsv_v_mean",
    "hsv_v_std",

    "lab_l_mean",
    "lab_a_mean",
    "lab_b_mean",

    "lab_l_std",
    "lab_a_std",
    "lab_b_std",

    "contrast_lab",
    "contrast_saturation",
    "contrast_brightness",

    "gray_mean",
    "gray_std",

    "entropy",
    "uniformity",

    "laplacian_variance",
    "gradient_mean",
    "gradient_std"
]


def calculate_asymmetry(
    mask
):

    ys, xs = np.where(
        mask > 0
    )

    if len(
        xs
    ) < 20:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    crop = mask[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    crop = cv2.resize(
        crop,
        (
            128,
            128
        ),
        interpolation=cv2.INTER_NEAREST
    )

    horizontal = cv2.flip(
        crop,
        1
    )

    vertical = cv2.flip(
        crop,
        0
    )

    union_h = np.logical_or(
        crop,
        horizontal
    ).sum()

    union_v = np.logical_or(
        crop,
        vertical
    ).sum()

    asym_h = float(
        np.logical_xor(
            crop,
            horizontal
        ).sum()
        /
        max(
            union_h,
            1
        )
    )

    asym_v = float(
        np.logical_xor(
            crop,
            vertical
        ).sum()
        /
        max(
            union_v,
            1
        )
    )

    return (
        asym_h,
        asym_v,
        float(
            (
                asym_h
                +
                asym_v
            )
            /
            2
        )
    )


def entropy_uniformity(
    values
):

    if len(
        values
    ) < 20:

        return (
            np.nan,
            np.nan
        )

    hist = np.bincount(
        values.astype(
            np.uint8
        ),
        minlength=256
    ).astype(
        np.float64
    )

    p = (
        hist
        /
        max(
            hist.sum(),
            1
        )
    )

    nz = p[
        p > 0
    ]

    entropy = float(
        -np.sum(
            nz
            *
            np.log2(
                nz
            )
        )
    )

    uniformity = float(
        np.sum(
            p ** 2
        )
    )

    return (
        entropy,
        uniformity
    )


def extract_features(
    rgb,
    mask
):

    result = {
        feature:
            np.nan
        for feature in FEATURE_COLUMNS
    }

    mask_bool = (
        mask > 0
    )

    if mask_bool.sum() < 50:

        return result

    h, w = mask.shape

    contours, _ = cv2.findContours(
        mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return result

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = float(
        cv2.contourArea(
            contour
        )
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True
        )
    )

    result[
        "area_relative"
    ] = (
        area
        /
        float(
            h * w
        )
    )

    result[
        "perimeter_normalized"
    ] = (
        perimeter
        /
        max(
            2.0
            *
            (
                h + w
            ),
            1.0
        )
    )

    circularity = (
        4.0
        *
        np.pi
        *
        area
        /
        max(
            perimeter ** 2,
            1e-8
        )
    )

    circularity = float(
        np.clip(
            circularity,
            0,
            1
        )
    )

    result[
        "circularity"
    ] = circularity

    result[
        "irregularity"
    ] = (
        1.0
        -
        circularity
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = max(
        float(
            cv2.contourArea(
                hull
            )
        ),
        1.0
    )

    result[
        "solidity"
    ] = (
        area
        /
        hull_area
    )

    x, y, bw, bh = cv2.boundingRect(
        contour
    )

    result[
        "extent"
    ] = (
        area
        /
        max(
            bw * bh,
            1
        )
    )

    result[
        "aspect_ratio"
    ] = (
        float(
            bw
        )
        /
        max(
            float(
                bh
            ),
            1.0
        )
    )

    if len(
        contour
    ) >= 5:

        ellipse = cv2.fitEllipse(
            contour
        )

        major = max(
            ellipse[
                1
            ]
        )

        minor = min(
            ellipse[
                1
            ]
        )

        if major > 0:

            result[
                "eccentricity"
            ] = float(
                np.sqrt(
                    max(
                        0.0,
                        1.0
                        -
                        (
                            minor
                            /
                            major
                        ) ** 2
                    )
                )
            )

    ah, av, am = calculate_asymmetry(
        mask
    )

    result[
        "asymmetry_horizontal"
    ] = ah

    result[
        "asymmetry_vertical"
    ] = av

    result[
        "asymmetry_mean"
    ] = am

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    lesion_rgb = rgb[
        mask_bool
    ].astype(
        np.float32
    )

    lesion_hsv = hsv[
        mask_bool
    ].astype(
        np.float32
    )

    lesion_lab = lab[
        mask_bool
    ].astype(
        np.float32
    )

    for channel, name in enumerate(
        [
            "r",
            "g",
            "b"
        ]
    ):

        result[
            f"rgb_{name}_mean"
        ] = float(
            lesion_rgb[
                :,
                channel
            ].mean()
        )

        result[
            f"rgb_{name}_std"
        ] = float(
            lesion_rgb[
                :,
                channel
            ].std()
        )

    result[
        "hsv_s_mean"
    ] = float(
        lesion_hsv[
            :,
            1
        ].mean()
    )

    result[
        "hsv_s_std"
    ] = float(
        lesion_hsv[
            :,
            1
        ].std()
    )

    result[
        "hsv_v_mean"
    ] = float(
        lesion_hsv[
            :,
            2
        ].mean()
    )

    result[
        "hsv_v_std"
    ] = float(
        lesion_hsv[
            :,
            2
        ].std()
    )

    for channel, name in enumerate(
        [
            "l",
            "a",
            "b"
        ]
    ):

        result[
            f"lab_{name}_mean"
        ] = float(
            lesion_lab[
                :,
                channel
            ].mean()
        )

        result[
            f"lab_{name}_std"
        ] = float(
            lesion_lab[
                :,
                channel
            ].std()
        )

    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            19,
            19
        )
    )

    dilated = cv2.dilate(
        mask.astype(
            np.uint8
        ),
        ring_kernel
    )

    ring = (
        (dilated > 0)
        &
        (~mask_bool)
    )

    if ring.sum() > 100:

        skin_lab = lab[
            ring
        ].astype(
            np.float32
        )

        skin_hsv = hsv[
            ring
        ].astype(
            np.float32
        )

        delta_lab = (
            lesion_lab.mean(
                axis=0
            )
            -
            skin_lab.mean(
                axis=0
            )
        )

        result[
            "contrast_lab"
        ] = float(
            np.linalg.norm(
                delta_lab
            )
        )

        result[
            "contrast_saturation"
        ] = float(
            lesion_hsv[
                :,
                1
            ].mean()
            -
            skin_hsv[
                :,
                1
            ].mean()
        )

        result[
            "contrast_brightness"
        ] = float(
            lesion_hsv[
                :,
                2
            ].mean()
            -
            skin_hsv[
                :,
                2
            ].mean()
        )

    lesion_gray = gray[
        mask_bool
    ]

    result[
        "gray_mean"
    ] = float(
        lesion_gray.mean()
    )

    result[
        "gray_std"
    ] = float(
        lesion_gray.std()
    )

    entropy, uniformity = entropy_uniformity(
        lesion_gray
    )

    result[
        "entropy"
    ] = entropy

    result[
        "uniformity"
    ] = uniformity

    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )

    result[
        "laplacian_variance"
    ] = float(
        lap[
            mask_bool
        ].var()
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        gx ** 2
        +
        gy ** 2
    )

    lesion_gradient = gradient[
        mask_bool
    ]

    result[
        "gradient_mean"
    ] = float(
        lesion_gradient.mean()
    )

    result[
        "gradient_std"
    ] = float(
        lesion_gradient.std()
    )

    return result


# =================================================================================================
# ROI J2.1
# =================================================================================================

def make_roi(
    rgb,
    mask,
    valid,
    roi_size=224,
    margin=0.16
):

    h, w = rgb.shape[
        :2
    ]

    if (
        not valid
        or
        mask.sum() < 50
    ):

        side = min(
            h,
            w
        )

        cx = w // 2
        cy = h // 2

        x1 = cx-side//2
        y1 = cy-side//2

        x2 = x1+side
        y2 = y1+side

        working_mask = np.zeros_like(
            mask
        )

    else:

        working_mask = mask

        ys, xs = np.where(
            mask > 0
        )

        xmin = int(
            xs.min()
        )

        xmax = int(
            xs.max()
        )

        ymin = int(
            ys.min()
        )

        ymax = int(
            ys.max()
        )

        bw = xmax-xmin+1
        bh = ymax-ymin+1

        side = int(
            np.ceil(
                max(
                    bw,
                    bh
                )
                *
                (
                    1.0
                    +
                    2.0
                    *
                    margin
                )
            )
        )

        side = max(
            side,
            32
        )

        cx = (
            xmin+xmax
        ) // 2

        cy = (
            ymin+ymax
        ) // 2

        x1 = cx-side//2
        y1 = cy-side//2

        x2 = x1+side
        y2 = y1+side

    pad_left = max(
        0,
        -x1
    )

    pad_top = max(
        0,
        -y1
    )

    pad_right = max(
        0,
        x2-w
    )

    pad_bottom = max(
        0,
        y2-h
    )

    if any(
        [
            pad_left,
            pad_top,
            pad_right,
            pad_bottom
        ]
    ):

        rgb = cv2.copyMakeBorder(
            rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101
        )

        working_mask = cv2.copyMakeBorder(
            working_mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0
        )

        x1 += pad_left
        x2 += pad_left

        y1 += pad_top
        y2 += pad_top

    roi = rgb[
        y1:y2,
        x1:x2
    ]

    roi_mask = working_mask[
        y1:y2,
        x1:x2
    ]

    roi = cv2.resize(
        roi,
        (
            roi_size,
            roi_size
        ),
        interpolation=cv2.INTER_AREA
    )

    roi_mask = cv2.resize(
        roi_mask,
        (
            roi_size,
            roi_size
        ),
        interpolation=cv2.INTER_NEAREST
    )

    roi_mask = (
        roi_mask > 0
    ).astype(
        np.uint8
    )

    return (
        roi,
        roi_mask
    )


# =================================================================================================
# MODELO
# =================================================================================================
