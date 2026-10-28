// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[YOLO11s_det] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Output tensor shape (batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // 84 = 4 box coords + 80 class scores
    // We need to identify which dim is the feature dim (84) and which is the
    // anchor/proposal count (8400).

    const unsigned dim0 = layer0.inferDims.d[0]; // 84
    const unsigned dim1 = layer0.inferDims.d[1]; // 8400

    // Identify feature dim vs count dim
    unsigned featureDim, countDim;
    bool transposed; // true if layout is [features, count] i.e. need transpose
    if (dim0 == (4u + numClasses)) {
        featureDim = dim0;
        countDim = dim1;
        transposed = true;
    } else if (dim1 == (4u + numClasses)) {
        featureDim = dim1;
        countDim = dim0;
        transposed = false;
    } else {
        fprintf(stderr, "[YOLO11s_det] ERROR: neither dim matches 4+numClasses (%u). "
                "d[0]=%u d[1]=%u\n", 4 + numClasses, dim0, dim1);
        return false;
    }

    const float iouThreshold = 0.45f;

    // Struct for candidate detections before NMS
    struct Detection {
        float cx, cy, w, h;
        float score;
        unsigned classId;
    };

    // Per-class buckets for NMS
    std::vector<std::vector<Detection>> perClassDets(numClasses);

    for (unsigned i = 0; i < countDim; ++i) {
        // Read cx, cy, w, h and class scores
        float cx_val, cy_val, w_val, h_val;
        if (transposed) {
            // data layout: [featureDim][countDim], row-major
            // element [f][i] = data[f * countDim + i]
            cx_val = data[0 * countDim + i];
            cy_val = data[1 * countDim + i];
            w_val  = data[2 * countDim + i];
            h_val  = data[3 * countDim + i];
        } else {
            // data layout: [countDim][featureDim]
            cx_val = data[i * featureDim + 0];
            cy_val = data[i * featureDim + 1];
            w_val  = data[i * featureDim + 2];
            h_val  = data[i * featureDim + 3];
        }

        // Find best class
        float maxScore = 0.0f;
        unsigned bestClass = 0;
        for (unsigned c = 0; c < numClasses; ++c) {
            float s;
            if (transposed) {
                s = data[(4 + c) * countDim + i];
            } else {
                s = data[i * featureDim + 4 + c];
            }
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        // Per-class threshold
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.35f;
        if (maxScore < thr) continue;

        // Filter to person class only (class_id == 0)
        if (bestClass != 0) continue;

        Detection det;
        det.cx = cx_val;
        det.cy = cy_val;
        det.w  = w_val;
        det.h  = h_val;
        det.score = maxScore;
        det.classId = bestClass;
        perClassDets[bestClass].push_back(det);
    }

    // Per-class NMS
    for (unsigned c = 0; c < numClasses; ++c) {
        auto& dets = perClassDets[c];
        if (dets.empty()) continue;

        // Sort by score descending
        std::sort(dets.begin(), dets.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.score > b.score;
                  });

        std::vector<bool> suppressed(dets.size(), false);
        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;

                // Compute IoU in network coords (cx/cy/w/h -> x1/y1/x2/y2)
                float ix1 = std::max(dets[i].cx - dets[i].w * 0.5f,
                                     dets[j].cx - dets[j].w * 0.5f);
                float iy1 = std::max(dets[i].cy - dets[i].h * 0.5f,
                                     dets[j].cy - dets[j].h * 0.5f);
                float ix2 = std::min(dets[i].cx + dets[i].w * 0.5f,
                                     dets[j].cx + dets[j].w * 0.5f);
                float iy2 = std::min(dets[i].cy + dets[i].h * 0.5f,
                                     dets[j].cy + dets[j].h * 0.5f);

                float interW = std::max(ix2 - ix1, 0.0f);
                float interH = std::max(iy2 - iy1, 0.0f);
                float inter = interW * interH;
                float areaI = dets[i].w * dets[i].h;
                float areaJ = dets[j].w * dets[j].h;
                float unionA = areaI + areaJ - inter;
                float iou = (unionA > 0.0f) ? (inter / unionA) : 0.0f;

                if (iou > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;
            const auto& d = dets[i];

            // Convert cx/cy/w/h to left/top/width/height in network coords
            float left   = d.cx - d.w * 0.5f;
            float top    = d.cy - d.h * 0.5f;
            float width  = d.w;
            float height = d.h;

            // Clamp to network input
            left   = std::max(left, 0.0f);
            top    = std::max(top, 0.0f);
            float right  = std::min(left + width, netW);
            float bottom = std::min(top + height, netH);
            width  = right - left;
            height = bottom - top;

            // Drop degenerate boxes
            if (width <= 1.0f || height <= 1.0f) continue;

            NvDsInferObjectDetectionInfo obj;
            obj.classId = d.classId;
            obj.left = left;
            obj.top = top;
            obj.width = width;
            obj.height = height;
            obj.detectionConfidence = d.score;
            objectList.push_back(obj);
        }
    }
    // ==== END GENERATED DECODE ====
    return true;
}