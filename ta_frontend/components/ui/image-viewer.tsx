import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Image, Modal, Pressable, StyleSheet, View } from 'react-native';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, { interpolate, runOnJS, useAnimatedStyle, useSharedValue, withTiming } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';

const MIN_VIEWER_SCALE = 1;
const MAX_VIEWER_SCALE = 4;
const SWIPE_CLOSE_THRESHOLD = 150;

type ImageViewerProps = {
  visible: boolean;
  uri: string | null;
  title: string;
  onClose: () => void;
};

function clampValue(value: number, min: number, max: number) {
  'worklet';

  return Math.min(Math.max(value, min), max);
}

function getMaxOffset(contentSize: number, viewportSize: number, scale: number) {
  'worklet';

  return Math.max(0, (contentSize * scale - viewportSize) / 2);
}

export function ImageViewer({ visible, uri, title, onClose }: ImageViewerProps) {
  const [imageNaturalSize, setImageNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  const scale = useSharedValue(1);
  const scaleOffset = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const translateXOffset = useSharedValue(0);
  const translateYOffset = useSharedValue(0);
  const closeSwipeY = useSharedValue(0);
  const closeOpacity = useSharedValue(1);
  const swipeStartScale = useSharedValue(1);

  const resetViewer = useCallback(() => {
    setImageNaturalSize(null);
    scale.value = 1;
    scaleOffset.value = 1;
    translateX.value = 0;
    translateY.value = 0;
    translateXOffset.value = 0;
    translateYOffset.value = 0;
    closeSwipeY.value = 0;
    closeOpacity.value = 1;
    swipeStartScale.value = 1;
  }, [
    closeOpacity,
    closeSwipeY,
    scale,
    scaleOffset,
    swipeStartScale,
    translateX,
    translateXOffset,
    translateY,
    translateYOffset,
  ]);

  useEffect(() => {
    resetViewer();
  }, [resetViewer, uri, visible]);

  const imageLayout = useMemo(() => {
    if (!imageNaturalSize || stageSize.width <= 0 || stageSize.height <= 0) {
      return null;
    }

    const fitScale = Math.min(
      stageSize.width / imageNaturalSize.width,
      stageSize.height / imageNaturalSize.height,
    );
    const displayedWidth = imageNaturalSize.width * fitScale;
    const displayedHeight = imageNaturalSize.height * fitScale;

    return {
      baseHeight: displayedHeight,
      baseWidth: displayedWidth,
      displayedHeight,
      displayedWidth,
    };
  }, [imageNaturalSize, stageSize.height, stageSize.width]);

  const closeViewer = useCallback(() => {
    resetViewer();
    onClose();
  }, [onClose, resetViewer]);

  const panGesture = useMemo(
    () =>
      Gesture.Pan()
        .onBegin(() => {
          translateXOffset.value = translateX.value;
          translateYOffset.value = translateY.value;
          swipeStartScale.value = scale.value;
        })
        .onUpdate((event) => {
          if (swipeStartScale.value === 1 && event.translationY > 0) {
            closeSwipeY.value = event.translationY;
            closeOpacity.value = interpolate(event.translationY, [0, SWIPE_CLOSE_THRESHOLD * 2], [1, 0.3]);
            scale.value = interpolate(event.translationY, [0, SWIPE_CLOSE_THRESHOLD * 2], [1, 0.85]);
            return;
          }

          const maxOffsetX = getMaxOffset(
            imageLayout?.displayedWidth ?? stageSize.width,
            stageSize.width,
            scale.value,
          );
          const maxOffsetY = getMaxOffset(
            imageLayout?.displayedHeight ?? stageSize.height,
            stageSize.height,
            scale.value,
          );

          translateX.value = clampValue(translateXOffset.value + event.translationX, -maxOffsetX, maxOffsetX);
          translateY.value = clampValue(translateYOffset.value + event.translationY, -maxOffsetY, maxOffsetY);
        })
        .onEnd(() => {
          if (closeSwipeY.value > 0) {
            if (closeSwipeY.value > SWIPE_CLOSE_THRESHOLD) {
              runOnJS(closeViewer)();
              return;
            }

            closeSwipeY.value = withTiming(0);
            closeOpacity.value = withTiming(1);
            scale.value = withTiming(1);
            return;
          }

          translateXOffset.value = translateX.value;
          translateYOffset.value = translateY.value;
        }),
    [
      closeSwipeY,
      closeOpacity,
      closeViewer,
      imageLayout?.displayedHeight,
      imageLayout?.displayedWidth,
      scale,
      stageSize.height,
      stageSize.width,
      swipeStartScale,
      translateX,
      translateXOffset,
      translateY,
      translateYOffset,
    ],
  );

  const pinchGesture = useMemo(
    () =>
      Gesture.Pinch()
        .onBegin(() => {
          scaleOffset.value = scale.value;
        })
        .onUpdate((event) => {
          const nextScale = clampValue(scaleOffset.value * event.scale, MIN_VIEWER_SCALE, MAX_VIEWER_SCALE);
          const maxOffsetX = getMaxOffset(
            imageLayout?.displayedWidth ?? stageSize.width,
            stageSize.width,
            nextScale,
          );
          const maxOffsetY = getMaxOffset(
            imageLayout?.displayedHeight ?? stageSize.height,
            stageSize.height,
            nextScale,
          );

          scale.value = nextScale;
          translateX.value = clampValue(translateX.value, -maxOffsetX, maxOffsetX);
          translateY.value = clampValue(translateY.value, -maxOffsetY, maxOffsetY);
        })
        .onEnd(() => {
          scaleOffset.value = scale.value;
          translateXOffset.value = translateX.value;
          translateYOffset.value = translateY.value;
        }),
    [
      imageLayout?.displayedHeight,
      imageLayout?.displayedWidth,
      scale,
      scaleOffset,
      stageSize.height,
      stageSize.width,
      translateX,
      translateXOffset,
      translateY,
      translateYOffset,
    ],
  );

  const viewerGesture = useMemo(() => Gesture.Simultaneous(panGesture, pinchGesture), [panGesture, pinchGesture]);

  const imageAnimatedStyle = useAnimatedStyle(() => ({
    opacity: closeOpacity.value,
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value + closeSwipeY.value },
      { scale: scale.value },
    ],
  }));

  const overlayAnimatedStyle = useAnimatedStyle(() => ({
    opacity: closeOpacity.value,
  }));

  return (
    <Modal animationType="fade" transparent visible={visible && uri !== null} onRequestClose={closeViewer}>
      <Animated.View style={[styles.viewerOverlay, overlayAnimatedStyle]}>
        <SafeAreaView style={styles.viewerContainer}>
          <Pressable style={styles.viewerCloseButton} onPress={closeViewer}>
            <Ionicons name="close" size={22} color="#FFFFFF" />
          </Pressable>

          <ThemedText style={styles.viewerTitle}>{title}</ThemedText>

          <GestureHandlerRootView style={styles.viewerStageRoot}>
            <View
              style={styles.viewerImageStage}
              onLayout={(event) => {
                const { width, height } = event.nativeEvent.layout;
                setStageSize((current) => {
                  if (current.width === width && current.height === height) {
                    return current;
                  }
                  return { width, height };
                });
              }}>
              <GestureDetector gesture={viewerGesture}>
                <Animated.View
                  style={[
                    styles.viewerImageShell,
                    imageLayout
                      ? {
                          height: imageLayout.baseHeight,
                          width: imageLayout.baseWidth,
                        }
                      : styles.viewerImageShellFallback,
                    imageAnimatedStyle,
                  ]}>
                  {uri ? (
                    <Image
                      source={{ uri }}
                      style={styles.viewerImage}
                      onLoad={(event) => {
                        const { width, height } = event.nativeEvent.source;
                        if (width && height) {
                          setImageNaturalSize({ width, height });
                        }
                      }}
                    />
                  ) : null}
                </Animated.View>
              </GestureDetector>
            </View>
          </GestureHandlerRootView>
        </SafeAreaView>
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  viewerCloseButton: {
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 22,
    height: 44,
    justifyContent: 'center',
    position: 'absolute',
    right: 16,
    top: 16,
    width: 44,
    zIndex: 10,
  },
  viewerContainer: {
    flex: 1,
  },
  viewerImage: {
    height: '100%',
    resizeMode: 'contain',
    width: '100%',
  },
  viewerImageShell: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  viewerImageShellFallback: {
    height: '100%',
    width: '100%',
  },
  viewerImageStage: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    overflow: 'hidden',
  },
  viewerOverlay: {
    backgroundColor: 'rgba(10, 16, 24, 0.96)',
    flex: 1,
  },
  viewerStageRoot: {
    flex: 1,
  },
  viewerTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    paddingHorizontal: 72,
    paddingTop: 24,
    textAlign: 'center',
  },
});
