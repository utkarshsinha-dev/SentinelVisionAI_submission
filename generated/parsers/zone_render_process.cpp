// nvcustomprocess CPU-mode user library (generated). Plain C++/OpenCV, no CUDA.
// Implements NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu*): the plugin
// maps the NV12 surface to host memory and calls this once per frame.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>      // cvtColor, resize, applyColorMap, putText, ...
#include <opencv2/dnn.hpp>          // cv::dnn::NMSBoxes (use for detection NMS)

#include "nvcustomprocess.h"        // CUDA-free ABI (forward-declares cudaStream_t)

#include <gst/gst.h>
#include "nvdsmeta.h"               // NvDsFrameMeta, NvDsObjectMeta, NvDsUserMeta
#include "gstnvdsinfer.h"           // NvDsInferTensorMeta, NvDsInferLayerInfo
#include "realsense_meta.h"         // RsMeta (installed in /usr/local/include)
#include "render_meta.h"            // publish what you decoded: render_meta::Builder

// ── BGR canvas -> NV12 planes (BT.601 limited range), respecting pitch ──────
static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

// ── NV12 surface -> BGR cv::Mat (when you draw on top of the live frame) ────
static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

// ── RealSense RsMeta lookup (null unless customrealsensesrc attach-nvds-meta) ─
static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// VERIFIED HELPERS — prefer these over hand-written equivalents (they fix the
// usual traps: wrong tensor pointer, batch-dim, NMSBoxesBatched, grid layout).
// ─────────────────────────────────────────────────────────────────────────────

// One model output: HOST data pointer + dims with a leading batch-1 stripped.
struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

// Collect EVERY frame-level output tensor as HOST copies (out_buf_ptrs_host — NEVER
// output_layers_info[i].buffer, which is device/unset). Pick one with find_tensor().
static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }   // strip batch
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

// Pick a tensor by its (batch-stripped) shape; -1 = "any". e.g.
// find_tensor(ts, 84, 6300) or find_tensor(ts, 18, 56, 56). Null if absent.
static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

// Class-aware NMS without NMSBoxesBatched (any OpenCV version). Parallel arrays;
// returns kept indices. Shifts classes apart so NMS is effectively per-class.
static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

// Draw `panel` into cell (r,c) of a rows×cols grid on `canvas` (auto-resized).
static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

// Min-max normalize a single-channel float map [h*w] and apply an OpenCV colormap
// (e.g. cv::COLORMAP_INFERNO). Returns an h×w BGR Mat. For depth/heatmaps.
static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[zone_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    // Model output tensor (config has output-tensor-meta=1; host copy ready):
    //   tm->out_buf_ptrs_host[i] -> float*, dims in tm->output_layers_info[i].inferDims
    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    // RealSense metadata (depth / IMU / intrinsics); null on non-RealSense pipelines.
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====
    (void) W; (void) H;  // pass-through
    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);
