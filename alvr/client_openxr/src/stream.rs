use crate::{
    graphics::{self, ProjectionLayerAlphaConfig, ProjectionLayerBuilder},
    interaction::{self, ButtonAction, InteractionContext, InteractionSourcesConfig},
};
use alvr_client_core::{
    ClientCoreContext,
    video_decoder::{self, VideoDecoderConfig, VideoDecoderSource},
};
use alvr_common::{
    DETACHED_CONTROLLER_LEFT_ID, DETACHED_CONTROLLER_RIGHT_ID, HAND_LEFT_ID, HAND_RIGHT_ID,
    HEAD_ID, LEFT_X_CLICK_ID, LEFT_Y_CLICK_ID, Pose, RelaxedAtomic, ViewParams,
    anyhow::Result,
    error, info, warn,
    glam::{UVec2, Vec2},
    parking_lot::RwLock,
};
use alvr_graphics::{GraphicsContext, StreamRenderer, StreamViewParams};
use alvr_packets::{RealTimeConfig, StreamConfig, TrackingData};
use alvr_session::{
    ClientsideFoveationConfig, ClientsideFoveationMode, ClientsidePostProcessingConfig, CodecType,
    FoveatedEncodingConfig, MediacodecProperty, PassthroughMode, UpscalingConfig,
};
use alvr_system_info::Platform;
use openxr as xr;
use std::{
    collections::HashMap,
    ptr,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const DECODER_MAX_TIMEOUT_MULTIPLIER: f32 = 0.8;

pub struct ParsedStreamConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub encoding_gamma: f32,
    pub enable_hdr: bool,
    pub passthrough: Option<PassthroughMode>,
    pub foveated_encoding_config: Option<FoveatedEncodingConfig>,
    pub clientside_foveation_config: Option<ClientsideFoveationConfig>,
    pub clientside_post_processing: Option<ClientsidePostProcessingConfig>,
    pub upscaling: Option<UpscalingConfig>,
    pub force_software_decoder: bool,
    pub max_buffering_frames: f32,
    pub buffering_history_weight: f32,
    pub decoder_options: Vec<(String, MediacodecProperty)>,
    pub interaction_sources: InteractionSourcesConfig,
    pub boundary_alert_enabled: bool,
}

impl ParsedStreamConfig {
    pub fn new(config: &StreamConfig) -> Self {
        Self {
            view_resolution: config.negotiated_config.view_resolution,
            refresh_rate_hint: config.negotiated_config.refresh_rate_hint,
            encoding_gamma: config.negotiated_config.encoding_gamma,
            enable_hdr: config.negotiated_config.enable_hdr,
            passthrough: config.settings.video.passthrough.as_option().cloned(),
            foveated_encoding_config: config
                .negotiated_config
                .enable_foveated_encoding
                .then(|| config.settings.video.foveated_encoding.as_option().cloned())
                .flatten(),
            clientside_foveation_config: config
                .settings
                .video
                .clientside_foveation
                .as_option()
                .cloned(),
            clientside_post_processing: config
                .settings
                .video
                .clientside_post_processing
                .as_option()
                .cloned(),
            upscaling: config.settings.video.upscaling.as_option().cloned(),
            force_software_decoder: config.settings.video.force_software_decoder,
            max_buffering_frames: config.settings.video.max_buffering_frames,
            buffering_history_weight: config.settings.video.buffering_history_weight,
            decoder_options: config.settings.video.mediacodec_extra_options.clone(),
            interaction_sources: InteractionSourcesConfig::new(config),
            boundary_alert_enabled: config.settings.audio.boundary_alert_enabled,
        }
    }
}

pub struct StreamContext {
    core_context: Arc<ClientCoreContext>,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_context: Arc<RwLock<InteractionContext>>,
    stage_reference_space: Arc<xr::Space>,
    view_reference_space: Arc<xr::Space>,
    swapchains: [xr::Swapchain<xr::OpenGlEs>; 2],
    last_good_view_params: [ViewParams; 2],
    input_thread: Option<JoinHandle<()>>,
    input_thread_running: Arc<RelaxedAtomic>,
    pub guardian_passthrough: Arc<RelaxedAtomic>,
    pub guardian_mode_idx: Arc<AtomicUsize>,
    config: ParsedStreamConfig,
    target_view_resolution: UVec2,
    renderer: StreamRenderer,
    decoder: Option<(VideoDecoderConfig, VideoDecoderSource)>,
    use_custom_reprojection: bool,
}

impl StreamContext {
    pub fn new(
        core_ctx: Arc<ClientCoreContext>,
        xr_session: xr::Session<xr::OpenGlEs>,
        gfx_ctx: Rc<GraphicsContext>,
        interaction_ctx: Arc<RwLock<InteractionContext>>,
        config: ParsedStreamConfig,
    ) -> StreamContext {
        interaction_ctx
            .write()
            .select_sources(&config.interaction_sources);

        let xr_exts = xr_session.instance().exts();

        if xr_exts.fb_display_refresh_rate.is_some() {
            xr_session
                .request_display_refresh_rate(config.refresh_rate_hint)
                .unwrap();
        }

        let foveation_profile = if let Some(config) = &config.clientside_foveation_config
            && xr_exts.fb_swapchain_update_state.is_some()
            && xr_exts.fb_foveation.is_some()
            && xr_exts.fb_foveation_configuration.is_some()
        {
            let level;
            let dynamic;
            match config.mode {
                ClientsideFoveationMode::Static { level: lvl } => {
                    level = lvl;
                    dynamic = false;
                }
                ClientsideFoveationMode::Dynamic { max_level } => {
                    level = max_level;
                    dynamic = true;
                }
            };

            xr_session
                .create_foveation_profile(Some(xr::FoveationLevelProfile {
                    level: xr::FoveationLevelFB::from_raw(level as i32),
                    vertical_offset: config.vertical_offset_deg,
                    dynamic: xr::FoveationDynamicFB::from_raw(dynamic as i32),
                }))
                .ok()
        } else {
            None
        };

        let target_view_resolution = alvr_graphics::compute_target_view_resolution(
            config.view_resolution,
            &config.upscaling,
        );
        let format = graphics::swapchain_format(&gfx_ctx, &xr_session, config.enable_hdr);

        let swapchains = [
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
        ];

        let renderer = StreamRenderer::new(
            gfx_ctx,
            config.view_resolution,
            target_view_resolution,
            [
                swapchains[0]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
                swapchains[1]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
            ],
            format,
            config.foveated_encoding_config.clone(),
            !((core_ctx.platform().is_pico()
                || (core_ctx.platform() == Platform::SamsungGalaxyXR))
                && config.enable_hdr),
            // TODO: Find a driver heuristic for the limited range bug instead?
            core_ctx.platform() != Platform::SamsungGalaxyXR && !config.enable_hdr,
            config.encoding_gamma,
            config.upscaling.clone(),
        );

        {
            let int_ctx = interaction_ctx.read();
            core_ctx.send_active_interaction_profile(
                *HAND_LEFT_ID,
                int_ctx.hands_interaction[0].controllers_profile_id,
                int_ctx.hands_interaction[0].input_ids.clone(),
            );
            core_ctx.send_active_interaction_profile(
                *HAND_RIGHT_ID,
                int_ctx.hands_interaction[1].controllers_profile_id,
                int_ctx.hands_interaction[1].input_ids.clone(),
            );
        }

        let input_thread_running = Arc::new(RelaxedAtomic::new(false));
        let guardian_passthrough = Arc::new(RelaxedAtomic::new(false));
        #[cfg(target_os = "android")]
        let guardian_mode_idx = Arc::new(AtomicUsize::new(DEFAULT_GUARDIAN_IDX));
        #[cfg(not(target_os = "android"))]
        let guardian_mode_idx = Arc::new(AtomicUsize::new(1));

        let stage_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        let view_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        let mut this = StreamContext {
            use_custom_reprojection: core_ctx.platform().is_yvr(),
            core_context: core_ctx,
            xr_session,
            interaction_context: interaction_ctx,
            stage_reference_space,
            view_reference_space,
            swapchains,
            last_good_view_params: [ViewParams::DUMMY; 2],
            input_thread: None,
            input_thread_running,
            guardian_passthrough,
            guardian_mode_idx,
            config,
            target_view_resolution,
            renderer,
            decoder: None,
        };

        this.update_reference_space();

        this
    }

    pub fn uses_passthrough(&self) -> bool {
        self.config.passthrough.is_some()
    }

    pub fn set_guardian_passthrough(&mut self, active: bool) {
        if active {
            if self.config.passthrough.is_none() {
                self.config.passthrough =
                    Some(PassthroughMode::Blend { premultiplied_alpha: false, threshold: 1.0 });
            }
        } else {
            if matches!(
                self.config.passthrough,
                Some(PassthroughMode::Blend { threshold, .. }) if (threshold - 1.0).abs() < f32::EPSILON
            ) {
                self.config.passthrough = None;
            }
        }
    }

    pub fn update_reference_space(&mut self) {
        self.input_thread_running.set(false);

        self.stage_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        self.view_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        #[cfg(target_os = "android")]
        {
            let idx = self.guardian_mode_idx.load(Ordering::Relaxed);
            let area = match GUARDIAN_CYCLE[idx] {
                None => Vec2::new(0.001, 0.001),
                Some(half) => Vec2::new(half * 2.0, half * 2.0),
            };
            match self
                .xr_session
                .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
            {
                Ok(Some(r)) => info!(
                    "[playspace] Quest reports {:.2}x{:.2} m; sending guardian config[{}]={:.0}x{:.0} m",
                    r.width, r.height, idx, area.x, area.y
                ),
                Ok(None) => info!(
                    "[playspace] Quest reports no bounds; sending guardian config[{}]={:.0}x{:.0} m",
                    idx, area.x, area.y
                ),
                Err(e) => warn!("[playspace] reference_space_bounds_rect failed: {e}"),
            }
            self.core_context.send_playspace(Some(area));
        }
        #[cfg(not(target_os = "android"))]
        self.core_context.send_playspace(Some(Vec2::new(20.0, 20.0)));

        if let Some(running) = self.input_thread.take() {
            running.join().ok();
        }

        self.input_thread_running.set(true);

        self.input_thread = Some(thread::spawn({
            let core_ctx = Arc::clone(&self.core_context);
            let xr_session = self.xr_session.clone();
            let interaction_ctx = Arc::clone(&self.interaction_context);
            let stage_reference_space = Arc::clone(&self.stage_reference_space);
            let view_reference_space = Arc::clone(&self.view_reference_space);
            let refresh_rate = self.config.refresh_rate_hint;
            let running = Arc::clone(&self.input_thread_running);
            let guardian_passthrough = Arc::clone(&self.guardian_passthrough);
            let guardian_mode_idx = Arc::clone(&self.guardian_mode_idx);
            move || {
                stream_input_loop(
                    &core_ctx,
                    xr_session,
                    &interaction_ctx,
                    &stage_reference_space,
                    &view_reference_space,
                    refresh_rate,
                    running,
                    guardian_passthrough,
                    guardian_mode_idx,
                )
            }
        }));
    }

    pub fn maybe_initialize_decoder(&mut self, codec: CodecType, config_nal: Vec<u8>) {
        let new_config = VideoDecoderConfig {
            codec,
            force_software_decoder: self.config.force_software_decoder,
            max_buffering_frames: self.config.max_buffering_frames,
            buffering_history_weight: self.config.buffering_history_weight,
            options: self.config.decoder_options.clone(),
            config_buffer: config_nal,
        };

        let maybe_config = if let Some((config, _)) = &self.decoder {
            (new_config != *config).then_some(new_config)
        } else {
            Some(new_config)
        };

        if let Some(config) = maybe_config {
            let (mut sink, source) = video_decoder::create_decoder(config.clone(), {
                let ctx = Arc::clone(&self.core_context);
                move |maybe_timestamp: Result<Duration>| match maybe_timestamp {
                    Ok(timestamp) => ctx.report_frame_decoded(timestamp),
                    Err(e) => ctx.report_fatal_decoder_error(&e.to_string()),
                }
            });
            self.decoder = Some((config, source));

            self.core_context.set_decoder_input_callback(Box::new(
                move |timestamp, buffer| -> bool { sink.push_nal(timestamp, buffer) },
            ));
        }
    }

    pub fn update_real_time_config(&mut self, config: &RealTimeConfig) {
        self.config.passthrough = config.passthrough.clone();
        self.config.clientside_post_processing = config.clientside_post_processing.clone();
    }

    pub fn render(
        &mut self,
        frame_interval: Duration,
        vsync_time: Duration,
    ) -> (ProjectionLayerBuilder<'_>, Duration) {
        let xr_vsync_time = xr::Time::from_nanos(vsync_time.as_nanos() as _);
        let frame_poll_deadline = Instant::now()
            + Duration::from_secs_f32(
                frame_interval.as_secs_f32() * DECODER_MAX_TIMEOUT_MULTIPLIER,
            );
        let mut frame_result = None;
        if let Some((_, source)) = &mut self.decoder {
            while frame_result.is_none() && Instant::now() < frame_poll_deadline {
                frame_result = source.get_frame();
                thread::sleep(Duration::from_micros(500));
            }
        }

        let (timestamp, view_params, buffer_ptr) =
            if let Some((timestamp, buffer_ptr)) = frame_result {
                let view_params = self.core_context.report_compositor_start(timestamp);

                self.last_good_view_params = view_params;

                (timestamp, view_params, buffer_ptr)
            } else {
                (vsync_time, self.last_good_view_params, ptr::null_mut())
            };

        let left_swapchain_idx = self.swapchains[0].acquire_image().unwrap();
        let right_swapchain_idx = self.swapchains[1].acquire_image().unwrap();

        self.swapchains[0]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();
        self.swapchains[1]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();

        let (flags, maybe_views) = self
            .xr_session
            .locate_views(
                xr::ViewConfigurationType::PRIMARY_STEREO,
                xr_vsync_time,
                &self.stage_reference_space,
            )
            .unwrap();

        let current_headset_views = if flags.contains(xr::ViewStateFlags::ORIENTATION_VALID) {
            maybe_views
        } else {
            vec![crate::default_view(), crate::default_view()]
        };

        // The poses and FoVs we received from the PC runtime, which may differ and/or include
        // altered FoVs based on settings and view conversions done for canting.
        let input_view_params = view_params;
        let mut output_view_params = input_view_params;
        // Avoid passing invalid timestamp to runtime.
        // `timestamp` is generally a current vsync time, but may be repeated if frames are
        // dropped. Some runtimes dislike it if the timestamp is repeated for too long, so after
        // one second we begin presenting a lagged vsync time instead.
        let mut openxr_display_time =
            Duration::max(timestamp, vsync_time.saturating_sub(Duration::from_secs(1)));

        // (shinyquagsire23) I don't entirely trust runtimes to implement CompositionLayerProjectionView
        // correctly, but if we do trust them, avoid doing rotation ourselves. Otherwise, rerender.
        // Ex: YVR/PFDMR has issues with aspect ratio mismatches and passthrough compositing.
        if self.use_custom_reprojection {
            output_view_params = [
                ViewParams {
                    pose: crate::from_xr_pose(current_headset_views[0].pose),
                    fov: crate::from_xr_fov(current_headset_views[0].fov),
                },
                ViewParams {
                    pose: crate::from_xr_pose(current_headset_views[1].pose),
                    fov: crate::from_xr_fov(current_headset_views[1].fov),
                },
            ];

            openxr_display_time = vsync_time;
        }

        self.renderer.render(
            buffer_ptr,
            [
                StreamViewParams {
                    swapchain_index: left_swapchain_idx,
                    input_view_params: input_view_params[0],
                    output_view_params: output_view_params[0],
                },
                StreamViewParams {
                    swapchain_index: right_swapchain_idx,
                    input_view_params: input_view_params[1],
                    output_view_params: output_view_params[1],
                },
            ],
            self.config.passthrough.as_ref(),
        );

        self.swapchains[0].release_image().unwrap();
        self.swapchains[1].release_image().unwrap();

        if !buffer_ptr.is_null()
            && let Some(xr_now) = crate::xr_runtime_now(self.xr_session.instance())
        {
            self.core_context.report_submit(
                timestamp,
                vsync_time.saturating_sub(Duration::from_nanos(xr_now.as_nanos() as u64)),
            );
        }

        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.target_view_resolution.x as _,
                height: self.target_view_resolution.y as _,
            },
        };

        let clientside_post_processing = self
            .xr_session
            .instance()
            .exts()
            .fb_composition_layer_settings
            .and(self.config.clientside_post_processing.clone());

        let layer = ProjectionLayerBuilder::new(
            &self.stage_reference_space,
            [
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(output_view_params[0].pose))
                    .fov(crate::to_xr_fov(output_view_params[0].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[0])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(output_view_params[1].pose))
                    .fov(crate::to_xr_fov(output_view_params[1].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[1])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
            ],
            self.config
                .passthrough
                .clone()
                .map(|mode| ProjectionLayerAlphaConfig {
                    premultiplied: matches!(
                        mode,
                        PassthroughMode::Blend {
                            premultiplied_alpha: true,
                            ..
                        } | PassthroughMode::RgbChromaKey(_)
                            | PassthroughMode::HsvChromaKey(_)
                    ),
                }),
            clientside_post_processing,
        );

        (layer, openxr_display_time)
    }
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        self.input_thread_running.set(false);
        self.input_thread.take().unwrap().join().ok();
    }
}

#[cfg(target_os = "android")]
struct BeepEnvelope<I> {
    inner: I,
    attack: u64,
    release: u64,
    total: u64,
    pos: u64,
}

#[cfg(target_os = "android")]
impl<I: rodio::Source<Item = f32>> BeepEnvelope<I> {
    fn new(inner: I, attack_ms: u64, total_ms: u64) -> Self {
        let sr = inner.sample_rate() as u64;
        let ch = inner.channels() as u64;
        let attack = sr * attack_ms * ch / 1000;
        let total = sr * total_ms * ch / 1000;
        Self { inner, attack, release: attack, total, pos: 0 }
    }
}

#[cfg(target_os = "android")]
impl<I: rodio::Source<Item = f32>> Iterator for BeepEnvelope<I> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        let attack_gain = if self.attack > 0 {
            (self.pos as f32 / self.attack as f32).min(1.0)
        } else {
            1.0
        };
        let release_gain = if self.release > 0 {
            (self.total.saturating_sub(self.pos) as f32 / self.release as f32).min(1.0)
        } else {
            1.0
        };
        let gain = attack_gain.min(release_gain);
        self.pos += 1;
        self.inner.next().map(|s| s * gain)
    }
}

#[cfg(target_os = "android")]
impl<I: rodio::Source<Item = f32>> rodio::Source for BeepEnvelope<I> {
    fn current_span_len(&self) -> Option<usize> { self.inner.current_span_len() }
    fn channels(&self) -> u16 { self.inner.channels() }
    fn sample_rate(&self) -> u32 { self.inner.sample_rate() }
    fn total_duration(&self) -> Option<Duration> { self.inner.total_duration() }
}

// Guardian area options cycled by short-pressing the left thumbstick while in config mode.
// None = disabled (no floor grid, no audio alert).
// Some(half) = half-extent in metres; full area = half*2 on each axis.
#[cfg(target_os = "android")]
const GUARDIAN_CYCLE: &[Option<f32>] = &[
    None,        // disabled
    Some(10.0),  // 20×20 m
    Some(7.5),   // 15×15 m  ← default (index 2)
    Some(5.0),   // 10×10 m
    Some(2.5),   //  5×5 m
    Some(1.25),  //  2.5×2.5 m
];
#[cfg(target_os = "android")]
const DEFAULT_GUARDIAN_IDX: usize = 2; // 15×15 m

#[cfg(target_os = "android")]
fn apply_guardian_mode(mode: Option<f32>, core_ctx: &ClientCoreContext, idx: usize) {
    match mode {
        None => {
            info!("[guardian] config[{}] = disabled", idx);
            core_ctx.send_playspace(Some(Vec2::new(0.001, 0.001)));
        }
        Some(half) => {
            let full = half * 2.0;
            info!("[guardian] config[{}] = {:.0}x{:.0} m", idx, full, full);
            core_ctx.send_playspace(Some(Vec2::new(full, full)));
        }
    }
}

fn stream_input_loop(
    core_ctx: &ClientCoreContext,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_ctx: &RwLock<InteractionContext>,
    stage_reference_space: &xr::Space,
    view_reference_space: &xr::Space,
    refresh_rate: f32,
    running: Arc<RelaxedAtomic>,
    #[cfg(target_os = "android")] guardian_passthrough: Arc<RelaxedAtomic>,
    #[cfg(not(target_os = "android"))] _guardian_passthrough: Arc<RelaxedAtomic>,
    #[cfg(target_os = "android")] guardian_mode_idx: Arc<AtomicUsize>,
    #[cfg(not(target_os = "android"))] _guardian_mode_idx: Arc<AtomicUsize>,
) {
    let mut last_controller_poses = [Pose::IDENTITY; 2];
    let mut last_palm_poses = [Pose::IDENTITY; 2];
    let mut last_view_params = [ViewParams::DUMMY; 2];

    // ── Guardian state ─────────────────────────────────────────────────────────
    // current_mode_idx indexes GUARDIAN_CYCLE; starts at 1 (20×20 m, enabled).
    // ui_in_config_mode: left-stick held 3 s enters config mode (passthrough on,
    //   4 s timeout).  Short tap in config mode cycles the mode.
    #[cfg(target_os = "android")]
    let alert_stream = {
        match rodio::OutputStreamBuilder::open_default_stream() {
            Ok(s) => { info!("[guardian] audio output stream opened"); Some(s) }
            Err(e) => { warn!("[guardian] failed to open audio output stream: {e}"); None }
        }
    };
    #[cfg(target_os = "android")]
    let mut current_mode_idx: usize = guardian_mode_idx.load(Ordering::Relaxed);
    #[cfg(target_os = "android")]
    {
        let half = GUARDIAN_CYCLE[current_mode_idx];
        let label = half.map_or("disabled".to_string(), |h| format!("{:.0}x{:.0} m", h*2.0, h*2.0));
        info!("[guardian] initialized: config[{}]={} (X+Y hold 1.5 s → config mode, X tap → cycle)", current_mode_idx, label);
    }
    // UI state: false = Normal, true = ConfigMode
    #[cfg(target_os = "android")]
    let mut ui_in_config_mode = false;
    #[cfg(target_os = "android")]
    let mut config_mode_timeout = Instant::now();
    // Entry: X+Y held 1.5 s simultaneously → config mode
    #[cfg(target_os = "android")]
    let mut both_held_since: Option<Instant> = None;
    #[cfg(target_os = "android")]
    let mut entry_consumed = false;
    // After entry, track X+Y both released before arming X for cycle
    #[cfg(target_os = "android")]
    let mut entry_released = false;
    // X tap while in config mode → cycle
    #[cfg(target_os = "android")]
    let mut x_armed = false;
    #[cfg(target_os = "android")]
    let mut next_beep_time = Instant::now();
    // Button edge-detection for debug logging (identify unknown button IDs)
    #[cfg(target_os = "android")]
    let mut prev_button_states: HashMap<u64, bool> = HashMap::new();

    let mut deadline = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / refresh_rate);
    while running.value() {
        let int_ctx_lock = interaction_ctx.read();
        let int_ctx = &*int_ctx_lock;
        // Streaming related inputs are updated here. Make sure every input poll is done in this
        // thread
        if let Err(e) = xr_session.sync_actions(&[(&int_ctx.action_set).into()]) {
            error!("{e}");
            return;
        }

        let Some(now) = crate::xr_runtime_now(xr_session.instance()).map(crate::from_xr_time)
        else {
            error!("Cannot poll tracking: invalid time");
            return;
        };

        let target_time = now + core_ctx.get_total_prediction_offset();

        let Some((head_motion, local_views)) = interaction::get_head_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            view_reference_space,
            now,
            target_time,
            &last_view_params,
        ) else {
            continue;
        };

        if let Some(views) = local_views {
            core_ctx.send_view_params(views);
            last_view_params = views;
        }

        // ── Guardian state machine ──────────────────────────────────────────────
        // Entry: X+Y held simultaneously 1.5 s → config mode (passthrough on).
        // Cycle: X tap while in config mode → next area option.
        // Exit: 4 s with no X tap → passthrough off, setting persists.
        #[cfg(target_os = "android")]
        {
            let get_bin = |id: &u64| -> bool {
                int_ctx
                    .button_actions
                    .get(id)
                    .and_then(|a| if let ButtonAction::Binary(a) = a {
                        a.state(&xr_session, xr::Path::NULL).ok()
                    } else { None })
                    .map_or(false, |s| s.current_state)
            };
            let x = get_bin(&LEFT_X_CLICK_ID);
            let y = get_bin(&LEFT_Y_CLICK_ID);
            let t = Instant::now();

            if !ui_in_config_mode {
                if x && y {
                    let held = both_held_since.get_or_insert(t);
                    if !entry_consumed && held.elapsed() >= Duration::from_millis(1500) {
                        ui_in_config_mode = true;
                        entry_consumed = true;
                        entry_released = false;
                        x_armed = false;
                        config_mode_timeout = t + Duration::from_secs(4);
                        guardian_passthrough.set(true);
                        info!("[guardian] entered config mode (X+Y held; passthrough on, 4 s timeout)");
                    }
                } else {
                    both_held_since = None;
                    entry_consumed = false;
                }
            } else {
                // Wait for X and Y to both be released after entry before arming X for cycle
                if !entry_released && !x && !y {
                    entry_released = true;
                    info!("[guardian] config mode: X+Y released, X tap to cycle options");
                }

                if entry_released {
                    if x && !x_armed {
                        x_armed = true;
                        config_mode_timeout = t + Duration::from_secs(4);
                    } else if !x && x_armed {
                        // X released → cycle
                        current_mode_idx = (current_mode_idx + 1) % GUARDIAN_CYCLE.len();
                        guardian_mode_idx.store(current_mode_idx, Ordering::Relaxed);
                        apply_guardian_mode(GUARDIAN_CYCLE[current_mode_idx], core_ctx, current_mode_idx);
                        x_armed = false;
                        config_mode_timeout = t + Duration::from_secs(4);
                    }
                }

                if t >= config_mode_timeout {
                    ui_in_config_mode = false;
                    x_armed = false;
                    entry_consumed = false;
                    both_held_since = None;
                    guardian_passthrough.set(false);
                    info!("[guardian] exited config mode (timeout)");
                }
            }
        }

        // ── Button debug logging (log every binary button on press edge) ────────
        #[cfg(target_os = "android")]
        for (id, action) in &int_ctx.button_actions {
            if let ButtonAction::Binary(a) = action {
                if let Ok(state) = a.state(&xr_session, xr::Path::NULL) {
                    let was = *prev_button_states.get(id).unwrap_or(&false);
                    if state.current_state && !was {
                        let path = alvr_common::BUTTON_INFO
                            .get(id)
                            .map(|i| i.path)
                            .unwrap_or("unknown");
                        info!("[btn] pressed: {}", path);
                    }
                    prev_button_states.insert(*id, state.current_state);
                }
            }
        }

        // ── Boundary proximity audio alert ──────────────────────────────────────
        #[cfg(target_os = "android")]
        if let Some(half) = GUARDIAN_CYCLE[current_mode_idx] {
            let t = Instant::now();
            let hx = head_motion.pose.position.x;
            let hz = head_motion.pose.position.z;
            let dx = (hx.abs() - half).max(0.0);
            let dz = (hz.abs() - half).max(0.0);
            let dist_outside = (dx * dx + dz * dz).sqrt();
            // 0.0 at boundary, 1.0 at 1.5 m outside
            let alert_level = (dist_outside / 1.5).min(1.0);
            if alert_level > 0.0 && t >= next_beep_time {
                if let Some(ref stream) = alert_stream {
                    use rodio::Source as _;
                    let raw = rodio::source::SineWave::new(880.0)
                        .take_duration(Duration::from_millis(100));
                    let beep = BeepEnvelope::new(raw, 10, 100).amplify(alert_level);
                    stream.mixer().add(beep);
                }
                // Interval: 3.0 s when barely outside → 0.3 s at 1.5 m+ outside
                let interval_secs = 3.0 - 2.7 * alert_level;
                let interval = Duration::from_secs_f32(interval_secs);
                next_beep_time = (next_beep_time + interval).max(t + interval);
            }
        }

        let mut device_motions = Vec::with_capacity(3);

        device_motions.push((*HEAD_ID, head_motion));

        let left_hand_data = crate::interaction::get_hand_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[0],
            &mut last_controller_poses[0],
            &mut last_palm_poses[0],
        );
        let right_hand_data = crate::interaction::get_hand_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[1],
            &mut last_controller_poses[1],
            &mut last_palm_poses[1],
        );

        // Note: When multimodal input is enabled, we are sure that when free hands are used
        // (not holding controllers) the controller data is None.
        if (int_ctx.multimodal_hands_enabled || left_hand_data.skeleton_joints.is_none())
            && let Some(motion) = left_hand_data.grip_motion
        {
            device_motions.push((*HAND_LEFT_ID, motion));
        }
        if (int_ctx.multimodal_hands_enabled || right_hand_data.skeleton_joints.is_none())
            && let Some(motion) = right_hand_data.grip_motion
        {
            device_motions.push((*HAND_RIGHT_ID, motion));
        }

        if int_ctx.multimodal_hands_enabled
            && let Some(detached_controller) = left_hand_data.detached_grip_motion
        {
            device_motions.push((*DETACHED_CONTROLLER_LEFT_ID, detached_controller));
        }
        if int_ctx.multimodal_hands_enabled
            && let Some(detached_controller) = right_hand_data.detached_grip_motion
        {
            device_motions.push((*DETACHED_CONTROLLER_RIGHT_ID, detached_controller));
        }

        let face = interaction::get_face_data(
            &xr_session,
            &int_ctx.face_sources,
            view_reference_space,
            now,
        );

        let body = int_ctx
            .body_source
            .as_ref()
            .and_then(|source| interaction::get_body_skeleton(source, stage_reference_space, now));

        if let Some(source) = &int_ctx.body_source {
            device_motions.append(&mut interaction::get_bd_motion_trackers(source, now));
        }

        drop(int_ctx_lock);

        let markers = interaction_ctx
            .write()
            .marker_spatial_context
            .as_mut()
            .and_then(|ctx| interaction::get_marker_poses(ctx, stage_reference_space, now))
            .unwrap_or_default();

        // Even though the server is already adding the motion-to-photon latency, here we use
        // target_time as the poll_timestamp to compensate for the fact that video frames are sent
        // with the poll timestamp instead of the vsync time. This is to ensure correctness when
        // submitting frames to OpenXR. This won't cause any desync with the server because no time
        // sync step is performed between client and server.
        core_ctx.send_tracking(TrackingData {
            poll_timestamp: target_time,
            device_motions,
            hand_skeletons: [
                left_hand_data.skeleton_joints,
                right_hand_data.skeleton_joints,
            ],
            face,
            body,
            markers,
        });

        let button_entries =
            interaction::update_buttons(&xr_session, &interaction_ctx.read().button_actions);
        if !button_entries.is_empty() {
            core_ctx.send_buttons(button_entries);
        }

        deadline += frame_interval / 3;
        thread::sleep(deadline.saturating_duration_since(Instant::now()));
    }
}
