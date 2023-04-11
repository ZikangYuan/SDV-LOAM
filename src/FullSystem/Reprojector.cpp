#include "utility"
#include "iostream"
#include <algorithm>
#include <stdexcept>
#include "FullSystem/Reprojector.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include "FullSystem/ResidualProjections.h"

namespace sdv_loam {

void Reprojector::getWarpMatrixAffine(
    const Eigen::Vector2d& px_ref,
    const Eigen::Vector3d& pt_ref,
    const SE3& T_cur_ref,
    Eigen::Matrix2d& A_cur_ref)
{
    const int halfpatch_size = 5;
    Eigen::Vector2d pixel_ref = px_ref;
    Eigen::Vector3d xyz_ref = pt_ref;
    Eigen::Vector2d pixel_du_ref = pixel_ref + Eigen::Vector2d(halfpatch_size, 0);
    Eigen::Vector2d pixel_dv_ref = pixel_ref + Eigen::Vector2d(0, halfpatch_size);
    Eigen::Vector3d xyz_du_ref = pixelFrame2UnitFrame(pixel_du_ref);
    Eigen::Vector3d xyz_dv_ref = pixelFrame2UnitFrame(pixel_dv_ref);
    xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
    SE3 T_cr = T_cur_ref;
    Eigen::Vector2d px_cur = pointRef2PixelCur(T_cr, xyz_ref);
    Eigen::Vector2d px_du = pointRef2PixelCur(T_cr, xyz_du_ref);
    Eigen::Vector2d px_dv = pointRef2PixelCur(T_cr, xyz_dv_ref);
    A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

int Reprojector::getBestSearchLevel(
    const Eigen::Matrix2d& A_cur_ref,
    const int max_level)
{
    int search_level = 0;
    double D = A_cur_ref.determinant();
    while(D > 3.0 && search_level < max_level)
    {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

void Reprojector::warpAffine(
    const Eigen::Matrix2d& A_cur_ref,
    const FrameHessian* img_ref,
    const Eigen::Vector2d& px_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
    const int patch_size = halfpatch_size*2 ;
    const Eigen::Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if(std::isnan(A_ref_cur(0,0)))
    {
        printf("Affine warp is NaN, probably camera has no translation\n");
        return;
    }

    uint8_t* patch_ptr = patch;
    const Eigen::Vector2f px_ref_pyr = px_ref.cast<float>();
    for (int y=0; y<patch_size; ++y)
    {
        for (int x=0; x<patch_size; ++x, ++patch_ptr)
        {
            Eigen::Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
            px_patch *= (1<<search_level);
            const Eigen::Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
            if (px[0] < 0 || px[1] < 0 || px[0] >= wG[0] - 1 || px[1] >= hG[0] - 1)
                *patch_ptr = 0;
            else
                *patch_ptr = (uint8_t) (getInterpolatedElement33(img_ref->dI, px[0], px[1], wG[0])[0]);
        }
    }
}

Reprojector::Reprojector(CalibHessian* Hcalib, FrameHessian* newframe, std::vector<FrameHessian*>& frameHessians) :
    Hcalib_(Hcalib), newframe_(newframe), frameHessians_(frameHessians)
{
    initializeGrid();
    K_ = Eigen::MatrixXd::Identity(3, 3);
    K_(0, 0) = Hcalib_->fxl(); K_(0, 2) = Hcalib_->cxl(); K_(1, 1) = Hcalib_->fyl(); K_(1, 2) = Hcalib_->cyl(); 
}

Reprojector::~Reprojector()
{
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
}

void Reprojector::initializeGrid()
{
    grid_.cell_size = 25;
    grid_.grid_n_cols = ceil(static_cast<double>(wG[0])/grid_.cell_size);
    grid_.grid_n_rows = ceil(static_cast<double>(hG[0])/grid_.cell_size);
    grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ c = new Cell; });
    grid_.cell_order.resize(grid_.cells.size());
    for(size_t i=0; i<grid_.cells.size(); ++i)
        grid_.cell_order[i] = i;
    random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end());
}

void Reprojector::resetGrid()
{
    n_matches_ = 0;
    n_trials_ = 0;
    std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
}

void Reprojector::reprojectMap(
    FrameHessian* frame,
    std::vector< std::pair<PointHessian*, Eigen::Vector2d> >& overlap_pts)
{
    resetGrid();

    std::list<std::pair<FrameHessian*, double>> close_kfs;

    for(int i = frameHessians_.size() - 1; i>=0 ; i--)
        close_kfs.push_back(std::make_pair(
            frameHessians_[i], (newframe_->shell->camToWorld.translation() - frameHessians_[i]->shell->camToWorld.translation()).norm()));

    close_kfs.sort(boost::bind(&std::pair<FrameHessian*, double>::second, _1) <
                  boost::bind(&std::pair<FrameHessian*, double>::second, _2));

    for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
        it_frame!=ite_frame; ++it_frame)
    {
        FrameHessian* ref_frame = it_frame->first;

        if(ref_frame == frame)
            continue;

        for(PointHessian* ph : ref_frame->pointHessians)
        {     
            if(reprojectPoint(frame, ph));
            {}
        }
    }

    for(int i=0; i<grid_.cells.size(); ++i)
    {
        if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, overlap_pts)){
            ++n_matches_;
        }
        if(n_matches_ > (int) (0.8 * setting_desiredImmatureDensity)){
            break;
        }
    }
}

void Reprojector::backprojectMap(FrameHessian* ref_frame,FrameHessian* frame,
    std::vector< std::pair<PointHessian*, Eigen::Vector2d> >& overlap_pts)
{
    backup = true;
    resetGrid();

    for(PointHessian* ph : frame->pointHessians)
    {     
        if(reprojectPoint(ref_frame, ph));
        {}
    }

    int num = 0;
    for(int i = 0; i < grid_.cells.size(); i++)
    {
        if(grid_.cells.at(i)->size() != 0)
            num++;
    }

    for(int i=0; i<grid_.cells.size(); ++i)
    {
        if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), ref_frame, overlap_pts)){
            ++n_matches_;
        }
        if(n_matches_ > (int) (0.8 * setting_desiredImmatureDensity)){
            break;
        }
    }
}

bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
    Vec2f lhColor = ((lhs.pt->host->dI)[(int)(lhs.pt->v * wG[0] + lhs.pt->u)]).tail<2>();
    Vec2f rhColor = ((rhs.pt->host->dI)[(int)(rhs.pt->v * wG[0] + rhs.pt->u)]).tail<2>();

    if(lhColor.norm() < rhColor.norm())
        return true;
    return false;
}

bool Reprojector::reprojectCell(Cell& cell, FrameHessian* frame, std::vector< std::pair<PointHessian*, Eigen::Vector2d> >& overlap_pts)
{
    cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
    Cell::iterator it=cell.begin();

    while(it!=cell.end())
    {
        if(it->pt->status != PointHessian::ACTIVE)
        {
            it = cell.erase(it);
            continue;
        }

        bool found_match = true;
        if(options_.find_match_direct)
            found_match = findMatchDirect(it->pt, frame, it->px);
        if(!found_match)
        {
            it = cell.erase(it);
            continue;
        }

        overlap_pts.push_back(std::pair<PointHessian*, Eigen::Vector2d>(it->pt, it->px));

        if(backup){
            it->pt->matcher.targetFrames.push_back(frame);
            it->pt->matcher.pxs.push_back(it->px);
            it->pt->matcher.frameIDs.push_back(frame->shell->id);
        }
        it = cell.erase(it);
        
        return true;      
    }

    return false;
}

bool Reprojector::findMatchDirect(
    PointHessian* pt,
    FrameHessian* cur_frame,
    Eigen::Vector2d& px_cur)
{
    FrameHessian* ref_ftr_ = NULL;

    if(frameHessians_.size()<=2){
        if(!backup)
            ref_ftr_ = frameHessians_[0];
        else if(backup && cur_frame == frameHessians_[0])
            ref_ftr_ = frameHessians_[1];
        else if(backup && cur_frame == frameHessians_[1])
            ref_ftr_ = frameHessians_[0];
    }
    else{
        ref_ftr_ = pt->host;
    }

    AffLight ref_aff_g2l = ref_ftr_->shell->aff_g2l;
    AffLight cur_aff_g2l = cur_frame->shell->aff_g2l;
    Vec2f affLL = AffLight::fromToVecExposure(ref_ftr_->ab_exposure, cur_frame->ab_exposure, ref_aff_g2l, cur_aff_g2l).cast<float>();

    Eigen::Vector3d ptWorld = pixelFrame2PointWorld(pt);
    Eigen::Vector3d ptRef = pointWorld2PointFrame(ref_ftr_, ptWorld);
    Eigen::Vector3d pixelRef = pointWorld2PixelFrame(ref_ftr_, ptWorld);
    Eigen::Vector2d px(pixelRef(0, 0), pixelRef(1, 0));

    if(!isInFrame(px.cast<int>(), halfpatch_size_+2))
        return false;

    Eigen::Matrix2d A_cur_ref_;
    getWarpMatrixAffine(px, ptRef, cur_frame->shell->camToWorld.inverse() * ref_ftr_->shell->camToWorld, A_cur_ref_);
    int search_level_ = getBestSearchLevel(A_cur_ref_, pyrLevelsUsed - 1);
    warpAffine(A_cur_ref_, ref_ftr_, px, search_level_, halfpatch_size_+1, patch_with_border_);
    createPatchFromPatchWithBorder();

    Eigen::Vector2d px_scaled(px_cur/(1<<search_level_));

    bool success = false;
    if(pt->type == PointHessian::EDGELET)
    {
        Eigen::Vector3d refDI = (ref_ftr_->dI[(int)(px(0, 0) + px(1, 0) * wG[0])]).cast<double>();
        Eigen::Vector2d refGrad = refDI.tail<2>();
        refGrad.normalize();
        Eigen::Vector2d dir_cur(A_cur_ref_ * refGrad);
        dir_cur.normalize();
        success = align1D(cur_frame->dIp[search_level_], dir_cur.cast<float>(),
            patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_, search_level_, affLL);
    }
    else
    {
        success = align2D(cur_frame->dIp[search_level_], patch_with_border_, patch_,
            options_.align_max_iter, px_scaled, search_level_, affLL);
    }
    px_cur = px_scaled * (1<<search_level_);
    return success;
}

bool Reprojector::getCloseViewObs(FrameHessian* frame, FrameHessian* &ftr, PointHessian* point)
{
    Eigen::Vector3d framepos = frame->shell->camToWorld.translation();
    Eigen::Vector3d pos_ = pixelFrame2PointWorld(point);
    Eigen::Vector3d obs_dir(framepos - pos_); obs_dir.normalize();

    FrameHessian* min_it = NULL;
    double min_cos_angle = 0;
    for(PointFrameResidual* r : point->residuals)
    {
        if(r->state_state != ResState::IN) continue;  

        FrameHessian* target = r->target;

        if(r->target == frame) continue;

        SE3 targetPose = r->target->shell->camToWorld;
        Eigen::Vector3d dir(targetPose.translation() - pos_); dir.normalize();
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = r->target;
        }
    }
    ftr = min_it;
    if(min_cos_angle < 0.5)
        return false;
    return true;
}

bool Reprojector::isInFrame(const Eigen::Vector2i& obs, int boundary)
{
    if(obs[0] >= boundary && obs[0] < wG[0] - boundary
        && obs[1] >= boundary && obs[1] < hG[0] - boundary)
        return true;
    return false;
}

void Reprojector::createPatchFromPatchWithBorder()
{
    uint8_t* ref_patch_ptr = patch_;
    for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_)
    {
        uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1;
        for(int x=0; x<patch_size_; ++x)
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
}

bool Reprojector::align1D(
    Eigen::Vector3f* cur_img,
    const Eigen::Vector2f& dir,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Eigen::Vector2d& cur_px_estimate,
    double& h_inv,
    int level,
    Vec2f& affLL)
{
    const int halfpatch_size_ = 4;
    const int patch_size = 8;
    const int patch_area = 64;
    bool converged=false;

    float __attribute__((__aligned__(16))) ref_patch_dv[patch_area];
    Eigen::Matrix2f H; H.setZero();

    const int ref_step = patch_size+2;
    float* it_dv = ref_patch_dv;
    for(int y=0; y<patch_size; ++y)
    {
        uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
        for(int x=0; x<patch_size; ++x, ++it, ++it_dv)
        {
            Eigen::Vector2f J;
            J[0] = 0.5*(dir[0]*(it[1] - it[-1]) + dir[1]*(it[ref_step] - it[-ref_step]));
            J[1] = 1;
            *it_dv = J[0];
            H += J*J.transpose();
        }
    }
    h_inv = 1.0/H(0,0)*patch_size*patch_size;
    Eigen::Matrix2f Hinv = H.inverse();
    float mean_diff = 0;

    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    const float min_update_squared = 0.03*0.03;
    const int cur_step = wG[level];
    float chi2 = 0;
    Eigen::Vector2f update; update.setZero();
    for(int iter = 0; iter<n_iter; ++iter)
    {
        int u_r = floor(u);
        int v_r = floor(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= wG[level]-halfpatch_size_ || v_r >= hG[level]-halfpatch_size_)
            break;

        if(std::isnan(u) || std::isnan(v))
            return false;

        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;

        uint8_t* it_ref = ref_patch;
        float* it_ref_dv = ref_patch_dv;
        float new_chi2 = 0.0;
        Eigen::Vector2f Jres; Jres.setZero();
        for(int y=0; y<patch_size; ++y)
        {
            Eigen::Vector3f* it = cur_img + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
            for(int x=0; x<patch_size; ++x, ++it, ++it_ref, ++it_ref_dv)
            {
                float search_pixel = wTL*(*it)[0] + wTR*(*(it+1))[0] + wBL*(*(it+cur_step))[0] + wBR*(*(it+cur_step+1))[0];
                float res = search_pixel - (float)(affLL[0] * (*it_ref) + affLL[1]) + mean_diff;
                Jres[0] -= res*(*it_ref_dv);
                Jres[1] -= res;
                new_chi2 += res*res;
            }
        }

        chi2 = new_chi2;
        update = Hinv * Jres;
        u += update[0]*dir[0];
        v += update[0]*dir[1];
        mean_diff += update[1];

#if SUBPIX_VERBOSE
        cout << "Iter " << iter << ":"
             << "\t u=" << u << ", v=" << v
             << "\t update = " << update[0] << ", " << update[1]
             << "\t new chi2 = " << new_chi2 << endl;
#endif

        if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
        {
#if SUBPIX_VERBOSE
            cout << "converged." << endl;
#endif
            converged=true;
            break;
        }
    }
    cur_px_estimate << u, v;
    return converged;
}

bool Reprojector::align2D(
    Eigen::Vector3f* cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Eigen::Vector2d& cur_px_estimate,
    int level, 
    Vec2f& affLL)
{
    const int halfpatch_size_ = 4;
    const int patch_size_ = 8;
    const int patch_area_ = 64;
    bool converged=false;

    float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
    float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
    Eigen::Matrix3f H; H.setZero();

    const int ref_step = patch_size_+2;
    float* it_dx = ref_patch_dx;
    float* it_dy = ref_patch_dy;
    for(int y=0; y<patch_size_; ++y)
    {
        uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
        for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
        {
            Eigen::Vector3f J;
            J[0] = 0.5 * (it[1] - it[-1]);
            J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
            J[2] = 1;
            *it_dx = J[0];
            *it_dy = J[1];
            H += J*J.transpose();
        }
    }
    Eigen::Matrix3f Hinv = H.inverse();
    float mean_diff = 0;

    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    const float min_update_squared = 0.03*0.03;
    const int cur_step = wG[level];
    //  float chi2 = 0;
    Eigen::Vector3f update; update.setZero();
    for(int iter = 0; iter<n_iter; ++iter)
    {
        int u_r = floor(u);
        int v_r = floor(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= wG[level]-halfpatch_size_ || v_r >= hG[level]-halfpatch_size_)
            break;

        if(std::isnan(u) || std::isnan(v))
            return false;

        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;

        uint8_t* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;

        Eigen::Vector3f Jres; Jres.setZero();
        for(int y=0; y<patch_size_; ++y)
        {
            Eigen::Vector3f* it = cur_img + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
            for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
            {
                float search_pixel = wTL*(*it)[0] + wTR*(*(it+1))[0] + wBL*(*(it+cur_step))[0] + wBR*(*(it+cur_step+1))[0];
                float res = search_pixel - (float)(affLL[0] * (*it_ref) + affLL[1]) + mean_diff;
                Jres[0] -= res*(*it_ref_dx);
                Jres[1] -= res*(*it_ref_dy);
                Jres[2] -= res;
            }
        }

        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];

#if SUBPIX_VERBOSE
        cout << "Iter " << iter << ":"
             << "\t u=" << u << ", v=" << v
             << "\t update = " << update[0] << ", " << update[1]
#endif

        if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
        {
#if SUBPIX_VERBOSE
            cout << "converged." << endl;
#endif
            converged=true;
            break;
        }
    }

    cur_px_estimate << u, v;
    return converged;
}

Eigen::Vector3d Reprojector::pixelFrame2UnitFrame(Eigen::Vector2d& px)
{
    Eigen::Vector3d pixelRef(px(0, 0), px(1, 0), 1.0);
    Eigen::Vector3d unitRef = K_.inverse() * pixelRef;
    return unitRef;
}

Eigen::Vector3d Reprojector::pixelFrame2PointWorld(PointHessian* point)
{
    Eigen::Vector3d pixelRef(point->u, point->v, 1.0);
    Eigen::Vector3d KiPixelRef = K_.inverse() * pixelRef;
    Eigen::Vector3d ptRef = KiPixelRef * (1/point->idepth);
    Eigen::Vector3d ptWorld = point->host->shell->camToWorld * ptRef;
    return ptWorld;
}

Eigen::Vector3d Reprojector::pointWorld2PixelFrame(FrameHessian* frame, Eigen::Vector3d& ptWorld)
{
    SE3 worldToCur = frame->shell->camToWorld.inverse();
    Eigen::Vector3d ptCur = worldToCur * ptWorld;
    ptCur[0] = ptCur[0] / ptCur[2]; ptCur[1] = ptCur[1] / ptCur[2]; ptCur[2] = ptCur[2] / ptCur[2];
    Eigen::Vector3d pixelCur = K_ * ptCur;
    return pixelCur;
}

Eigen::Vector3d Reprojector::pointWorld2PointFrame(FrameHessian* frame, Eigen::Vector3d& ptWorld)
{
    SE3 worldToCur = frame->shell->camToWorld.inverse();
    Eigen::Vector3d ptCur = worldToCur * ptWorld;
    return ptCur;
}


Eigen::Vector3d Reprojector::pointRef2PointCur(SE3& T_cur_ref, Eigen::Vector3d& ptRef)
{
    Eigen::Vector3d ptCur = T_cur_ref * ptRef;
    return ptCur;
}

Eigen::Vector2d Reprojector::pointRef2PixelCur(SE3& T_cur_ref, Eigen::Vector3d& ptRef)
{
    Eigen::Vector3d ptCur = T_cur_ref * ptRef;
    ptCur(0, 0) = ptCur(0, 0) / ptCur(2, 0); ptCur(1, 0) = ptCur(1, 0) / ptCur(2, 0); ptCur(2, 0) = ptCur(2, 0) / ptCur(2, 0);
    Eigen::Vector3d pixelCur = K_ * ptCur;
    Eigen::Vector2d pxCur(pixelCur(0, 0), pixelCur(1, 0));
    return pxCur;
}

bool Reprojector::reprojectPoint(FrameHessian* frame, PointHessian* point)
{
    Eigen::Vector3d ptWorld = pixelFrame2PointWorld(point);
    Eigen::Vector3d pixelCur = pointWorld2PixelFrame(frame, ptWorld);

    Eigen::Vector2d px(pixelCur(0, 0), pixelCur(1, 0));

    if(isInFrame(px.cast<int>(), 8))
    {
        const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                    + static_cast<int>(px[0]/grid_.cell_size);
        grid_.cells.at(k)->push_back(Candidate(point, px));
        return true;
    }
    return false;
}

}
