using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

public class LoadVideoClip : MonoBehaviour
{
    GameObject screen;
    GameObject player;
    // public string videoPath;
    public RenderTexture renderTextureTemplate;
    public Material matTemplate;
    public void SetVideoPath(string videoPath)
    {
        screen = transform.GetChild(0).gameObject;
        player = transform.GetChild(1).gameObject;

        RenderTexture videoTexture = new RenderTexture(renderTextureTemplate);
        Material mat = new Material(matTemplate);
        mat.SetTexture("_MainTex", videoTexture);
        // mat.SetTexture("_EmissionMap", videoTexture);
        screen.GetComponent<MeshRenderer>().material = mat;

        VideoPlayer videoPlayer = player.GetComponent<VideoPlayer>();
        VideoClip c = Resources.Load<VideoClip>(videoPath) as VideoClip;

        videoPlayer.clip = c;
        videoPlayer.targetTexture = videoTexture;
    }
}
