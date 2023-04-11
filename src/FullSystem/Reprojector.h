#pragma once

#include "util/NumType.h"
#include "util/globalCalib.h"
#include "vector"
#include "list"
#include <math.h>
#include "util/settings.h"

namespace sdv_loam
{
struct CalibHessian;
struct FrameHessian;
struct PointHessian;
struct PointFrameResidual;

class Reprojector
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// Reprojector config parameters
    struct Options {
        size_t max_n_kfs;   //!< max number of keyframes to reproject from
        bool find_match_direct;
        int align_max_iter;
        Options()
        : max_n_kfs(10),
        find_match_direct(true), 
        align_max_iter(10)
        {}
    } options_;

    size_t n_matches_;
    size_t n_trials_;

    Reprojector(CalibHessian* Hcalib, FrameHessian* newframe, std::vector<FrameHessian*>& frameHessians);


    ~Reprojector();

    /// Project points from the map into the image. First finds keyframes with
    /// overlapping field of view and projects only those map-points.
    void reprojectMap(
        FrameHessian* frame,
        std::vector< std::pair<PointHessian*, Eigen::Vector2d> >& overlap_pts);
    void backprojectMap(FrameHessian* ref_frame,FrameHessian* frame,
        std::vector< std::pair<PointHessian*, Eigen::Vector2d> >& overlap_pts);

    Eigen::Vector3d pixelFrame2UnitFrame(Eigen::Vector2d& px);
    Eigen::Vector3d pixelFrame2PointWorld(PointHessian* point);
    Eigen::Vector3d pointWorld2PixelFrame(FrameHessian* frame, Eigen::Vector3d& ptWorld);
    Eigen::Vector3d pointWorld2PointFrame(FrameHessian* frame, Eigen::Vector3d& ptWorld);
    Eigen::Vector3d pointRef2PointCur(SE3& T_cur_ref, Eigen::Vector3d& ptRef);
    Eigen::Vector2d pointRef2PixelCur(SE3& T_cur_ref, Eigen::Vector3d& ptRef);

private:

    /// A candidate is a point that projects into the image plane and for which we
    /// will search a maching feature in the image.
    struct Candidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PointHessian* pt;       //!< 3D point.
        Eigen::Vector2d px;     //!< projected 2D pixel location.
        Candidate(PointHessian* pt, Eigen::Vector2d& px) : pt(pt), px(px) {}
    };
    typedef std::list<Candidate > Cell;
    typedef std::vector<Cell*> CandidateGrid;

    bool backup = false;

    /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
    struct Grid
    {
        CandidateGrid cells;
        std::vector<int> cell_order;
        int cell_size;
        int grid_n_cols;
        int grid_n_rows;
    };

    Grid grid_;
    CalibHessian* Hcalib_;
    Eigen::Matrix3d K_ ;
    FrameHessian* newframe_;
    std::vector<FrameHessian*>& frameHessians_;

    static const int halfpatch_size_ = 4;
    static const int patch_size_ = 8;
    uint8_t patch_[patch_size_*patch_size_] __attribute__ ((aligned (16)));
    uint8_t patch_with_border_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));
    double h_inv_;

    static bool pointQualityComparator(Candidate& lhs, Candidate& rhs);
    void initializeGrid();
    void resetGrid();
    bool reprojectCell(Cell& cell, FrameHessian*  frame, std::vector< std::pair<PointHessian*, Eigen::Vector2d> >& overlap_pts);
    bool reprojectPoint(FrameHessian* frame, PointHessian* point);

    void getWarpMatrixAffine(const Eigen::Vector2d& px_ref, const Eigen::Vector3d& pt_ref, const SE3& T_cur_ref, Eigen::Matrix2d& A_cur_ref);
    int getBestSearchLevel(const Eigen::Matrix2d& A_cur_ref, const int max_level);
    void warpAffine(const Eigen::Matrix2d& A_cur_ref, const FrameHessian* img_ref, const Eigen::Vector2d& px_ref, const int search_level,
        const int halfpatch_size, uint8_t* patch);

    bool findMatchDirect(PointHessian* pt, FrameHessian* cur_frame, Eigen::Vector2d& px_cur);
    bool getCloseViewObs(FrameHessian* frame, FrameHessian* &ftr, PointHessian* point);
    bool isInFrame(const Eigen::Vector2i& obs, int boundary=0);
    void createPatchFromPatchWithBorder();
    bool align1D(Eigen::Vector3f* cur_img, const Eigen::Vector2f& dir, uint8_t* ref_patch_with_border,
        uint8_t* ref_patch, const int n_iter, Eigen::Vector2d& cur_px_estimate, double& h_inv, int level, Vec2f& affLL);
    bool align2D(Eigen::Vector3f* cur_img, uint8_t* ref_patch_with_border, uint8_t* ref_patch,
        const int n_iter, Eigen::Vector2d& cur_px_estimate, int level, Vec2f& affLL);
};
}