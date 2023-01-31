using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseOverTest : MonoBehaviour {
    void OnMouseEnter () {
        transform.GetChild(0).gameObject.SetActive(true);
        transform.GetChild(1).gameObject.SetActive(true);
    }

    void OnMouseExit()
    {
        transform.GetChild(0).gameObject.SetActive(false);
        transform.GetChild(1).gameObject.SetActive(false);
    }
}