package org.openvino.java.utils;


import java.util.List;

public class ListUtils {

//    public T List arrayToList(Class cls,) {
//
//    }

    public static  <T> T listToArray(Class cls, List data) {
        if (cls.getName().equals("[F")) {

        } else if (cls.getName().equals("[Ljava.lang.String;")) {

        } else if (cls.getName().equals("[B")) {

        } else if (cls.getName().equals("[C")) {

        } else if (cls.getName().equals("[D")) {

        } else if (cls.getName().equals("[S")) {

        } else if (cls.getName().equals("[L")) {

        } else if (cls.getName().equals("[I")) {
            int[] result = new int[data.size()];
            for (int i = 0; i < result.length;i++) {
                Object d = data.get(i);
                result[i] = d instanceof Integer ? (Integer)d : (int)d;
            }
        }
        return null;
    }
}
