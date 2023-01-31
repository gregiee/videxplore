using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Move : MonoBehaviour
{
    float moveSpeed = 0.01f;

    // Update is called once per frame
    void Update()
    {
        Vector3 deltaPos = Vector3.zero;

        if (Input.GetKey(KeyCode.W))
        {
            deltaPos += transform.up;
        }
        if (Input.GetKey(KeyCode.S))
        {
            deltaPos -= transform.up;
        }
        if (Input.GetKey(KeyCode.A))
        {
            deltaPos -= transform.right;
        }
        if (Input.GetKey(KeyCode.D))
        {
            deltaPos += transform.right;
        }

        transform.position += deltaPos * moveSpeed;
    }
}
