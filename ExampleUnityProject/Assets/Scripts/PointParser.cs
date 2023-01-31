using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

using TMPro;

public class PointParser : MonoBehaviour
{
    [Header("Visualization Metadata")]
    public string pointFileName;
    public bool showVideos;
    public string videoPathsFileName;
    public bool showDescriptions;
    public string descriptionFileName;

    [Space]
    [Header("Prefab Configurations")]
    public GameObject pointInstancePrefab;
    
    List<Vector3> points = new List<Vector3>();

    void Start()
    {
        TextAsset pointFile = Resources.Load<TextAsset>("Texts/" + pointFileName);
        TextAsset videoPathsFile = Resources.Load<TextAsset>("Texts/" + videoPathsFileName);
        TextAsset descriptionFile = Resources.Load<TextAsset>("Texts/" + descriptionFileName);

        string[] pointText = pointFile.text.Split("\r\n");
        string[] videoPathsText = videoPathsFile.text.Split("\r\n");
        string[] descriptionText = descriptionFile.text.Split("\r\n");

        for (int i = 0; i < pointText.Length; i++)
        {
            string[] line = pointText[i].Split(",");
            Vector3 point = new Vector3(float.Parse(line[0]), 0.0f, float.Parse(line[1]));
            points.Add(point);
            GameObject pointInstance = Instantiate(pointInstancePrefab, point, pointInstancePrefab.transform.rotation);

            GameObject descInstance = pointInstance.transform.GetChild(0).gameObject;
            descInstance.GetComponent<TMP_Text>().text = descriptionText[i];
            
            string videoPath = AssembleVideoPath(videoPathsText[i]);
            GameObject videoInstance = pointInstance.transform.GetChild(1).gameObject;
            videoInstance.GetComponent<LoadVideoClip>().SetVideoPath(videoPath);
        }
    }

    string AssembleVideoPath(string videoPath)
    {
        string fileName = System.IO.Path.GetFileName(videoPath);
        int suffixIndex = fileName.LastIndexOf(".");
        return "Videos/" + fileName.Substring(0, suffixIndex);
    }
}
