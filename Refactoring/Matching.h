

const int MAX_INT = 1<<30;

/*
Matching Parameters:

    In funcion "DeterminingFavorites" the delta parameter stands for the valid search space for finding potential matches. (in number of pixels)
    In funcion "SumofAbsoluteDifferences" the w parameter stands for the neighborhood used for determining the affinity response between feature points. (in number of pixels).
    Main function: harrisFeatureMatcherMCC  (harris Feature Matcher with mutual consistency check).
    To use this module look at the Main function.
*/

/*Matching*/

int SumofAbsoluteDifferences(Mat img1, Mat img2, pair<int,int> f1, pair<int,int> f2, int w = 10){
    if ( (img1.rows != img2.rows ) || (img1.cols != img2.cols)){
        printf("Las dimensiones de las imagenes no coinciden");
        return 0;
    }
    int limitx = img1.rows;
    int limity = img1.cols;
    int response = 0;
    for(int i = -w; i <= w; i++){
        for(int j = -w; j <= w; j++){
            int nx1 = f1.first + i;
            int ny1 = f1.second + j;
            int nx2 = f2.first + i;
            int ny2 = f2.second + j;
            bool c1 = (nx1 < 0 || ny1 < 0 || nx2 < 0 || ny2 < 0);
            bool c2 = (nx1 >= limitx || ny1 >= limity || nx2 >= limitx || ny2 >= limity);
            if (c1 || c2) continue;
            response += abs(img1.at<int>(nx1,ny1) - img2.at<int>(nx2,ny2));   
        }
    }
    return response;
}

     

void DeterminingFavorites(Mat img1, Mat img2, vector< pair<int,int> > &f1, vector< pair<int,int> > &f2, int *favorites, int delta = 15){
    for(int i = 0; i < f1.size(); i++){
        pair<int, int> current1 = f1[i];
        int menor = MAX_INT;
        int idMenor = -1;
        for(int j = 0; j < f2.size(); j++){
            pair<int, int> current2 = f2[j];
            int dx = abs(current1.first - current2.first);
            int dy = abs(current1.second - current2.second);
            if(dx > delta || dy > delta) continue;
            int similarity = SumofAbsoluteDifferences(img1, img2, current1, current2);
            if(similarity < menor){
                menor = similarity;
                idMenor = j;
            } 
        }
        favorites[i] = idMenor;
    }
    return;
}


vector<pair<int,int> > harrisFeatureMatcherMCC(Mat img1, Mat img2, vector< pair<int,int> > featuresImg1, vector< pair<int,int> > featuresImg2){
    int favoritesfromimg1[featuresImg1.size()]; // en la posicion i se guarda el favorito de la característica i de la imagen 1.
    int favoritesfromimg2[featuresImg2.size()]; // en la posicion i se guarda el favorito de la característica i de la imagen 2.
    vector< pair<int,int> > correspondences;
    DeterminingFavorites(img1, img2, featuresImg1, featuresImg2, favoritesfromimg1);
    DeterminingFavorites(img2, img1, featuresImg2, featuresImg1, favoritesfromimg2);
    for(int i = 0; i < featuresImg1.size(); ++i){
        if(favoritesfromimg2[favoritesfromimg1[i]] == i) correspondences.push_back(make_pair(i, favoritesfromimg1[i]));
    }
    return correspondences;
}

/*end Matching*/
