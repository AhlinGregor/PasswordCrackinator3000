package org.example;


public class CUDASolution {
    static {
        System.load("D:/FAKS/Prog3/PasswordCrackinator3000/build/Release/cudaHasher.dll");
    }

    public static native String nativeBruteForce(String charset, int length, byte[] hashBytes, int mode);

    private final static String smallAlpha = "zyxwvutsrqponmlkjihgfedcba";
    private final static String bigAlpha = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
    private final static char[] nonAlphabeticalCharacters = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  // Digits
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',   // Symbols
            '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|',  // Brackets and slashes
            ';', ':', '\'', '\"', ',', '<', '.', '>', '/', '?', // Punctuation
            '`', '~'                                           // Miscellaneous
    };
    private final static String nonAlpha = new String(nonAlphabeticalCharacters);

    /**
     * Method to decide if we're computing an MD5 or an SHA-256 hash
     * @param hash Password hash given as a user input
     * @param opt Options for lowercase, uppercase and special characters (including numbers)
     * @param length Length of the password
     * @return  A password that if hashed with the correct algorithm will return the parameter "hash" or null if the password is not found
     */
    public static String computeDizShiz(String hash, int opt, int length) {
        String charset = getCharacterSet(opt);
        byte[] target = hexToBytes(hash);
        int mode = isValidMD5(hash) ? 0 : isValidSHA(hash) ? 1 : -1;
        if (mode < 0) return null;
        return new CUDASolution().nativeBruteForce(charset, length, target, mode);
    }

//    private static String findMatchingPermutationSHA(String charset, int length, String targetHash) {
//        String found = null;
//        byte[] target = hexToBytes(targetHash);
//        found = new CUDASolution().nativeBruteForce(charset, length, target, 0);
//        return found;
//    }
//
//    private static String findMatchingPermutationMD(String charset, int length, String targetHash) {
//        String found = null;
//        byte[] target = hexToBytes(targetHash);
//        found = new CUDASolution().nativeBruteForce(charset, length, target, 1);
//        return found;
//    }

    private static byte[] hexToBytes(String s) {
        int len = s.length();
        byte[] out = new byte[len / 2];
        for (int i = 0; i < len; i += 2)
            out[i / 2] = (byte)((Character.digit(s.charAt(i), 16) << 4)
                    + Character.digit(s.charAt(i+1), 16));
        return out;
    }

    /**
     * Method to validate if the hash is in line with the SHA-256 requirements
     * @param input User-given hash
     * @return true if it is a valid hash, false otherwise
     */
    private static boolean isValidSHA(String input) {
        return input != null && input.matches("^[a-fA-F0-9]{64}$"); //"Yayy! Regex!" said he, sarcastically
    }

    /**
     * Method that calculates all possible combinations for a given length and charset
     * @param charsetLength All possible characters
     * @param maxLength The password length
     * @return a long integer that represents the number of possible -combinations
     */
    public static long calculateTotalCombinations(int charsetLength, int maxLength) {

        return (long) Math.pow(charsetLength, maxLength);
    }

    /**
     * Method to validate if the hash is in line with the MD5 requirements
     * @param input User-given hash
     * @return true if it is a valid hash, false otherwise
     */
    private static boolean isValidMD5(String input) {
        if (input == null || input.length() != 32) {
            return false;
        }

        return input.matches("[a-fA-F0-9]{32}"); // The bane of my existence yet again
    }

    /**
     * Method that generates a character set of available character as specified by the user
     * @param opt Integer representation of options
     * @return A String with all possible characters
     */
    public static String getCharacterSet(int opt) {
        switch (opt) {
            case 1 : return CUDASolution.smallAlpha;
            case 2 : return CUDASolution.bigAlpha;
            case 3 : return CUDASolution.smallAlpha + CUDASolution.bigAlpha;
            case 4 : return CUDASolution.nonAlpha;
            case 5 : return CUDASolution.smallAlpha + CUDASolution.nonAlpha;
            case 6 : return CUDASolution.bigAlpha + CUDASolution.nonAlpha;
            default : return CUDASolution.smallAlpha + CUDASolution.bigAlpha + CUDASolution.nonAlpha;
        }
    }
}
