# -*- coding: utf-8 -*-
import arcpy, os, time

# print "==============================================="
# arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis/home")
# # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com:6443/arcgis/manager")
# print arcpy.GetMessages()
# print arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis/home")
# print "登录门户成功"
# print "==============================================="
# 将指定目录下所有的.mxd文档发布为地图服务
# folder：包含mxd文档的文件夹路径
# serviceDir：服务目录URL，例如http://localhost/arcgis/rest/services
# serviceFolder：服务所在文件夹，如果为空，则表示根目录
def PublishAll(folder, serviceDir, serviceFolder):
    print "检查文件夹路径……"
    if os.path.isdir(folder) == False:
        print "输入的文件夹路径无效！"
        return
    print "遍历文件夹……"
    files = os.listdir(folder)
    for f in files:
        if f.endswith(".mxd"):
            mxdPath = os.path.join(folder, f)
            print "publishing: " + f
            PublishMxd(f, mxdPath, serviceDir, serviceFolder)
        else:
            continue

# 将mxd文档发布为服务：1.将mxd转为msd；2.分析msd；3.发布msd
def PublishMxd(mxdName, mxdPath, serviceDir, serviceFolder):
    # print "==============================================="
    # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis/home")
    # # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com:6443/arcgis/manager")
    # print arcpy.GetMessages()
    # print arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis/home")
    # print "登录门户成功"
    # print "==============================================="

    # 检查mxd文件是否存在
    print "检查文件路径……"
    if os.path.exists(mxdPath) == False:
        print "指定路径的mxd文档不存在！"
        return

    # 打开mxd文档
    try:
        print "正在打开mxd文档……"
        mxd = arcpy.mapping.MapDocument(mxdPath)
    except Exception, e:
        print "open mxd error: ", e
        return
    else:
        print "mxd文档打开成功……"

    # 获取默认的数据框
    # 构造sddraft文档名称
    sddraft = mxdPath.replace(".mxd", ".sddraft")
    service = mxdName.replace(".mxd", "")
    sd = mxdPath.replace(".mxd", ".sd")
    con = r'C:\Users\pc\AppData\Roaming\Esri\Desktop10.5\ArcCatalog\arcgis on arcgis.ygwl.com_6080 (发布者) (2).ags'
    copy_data_to_server = True
    # 正在将mxd文档转换为sddraft文档……"
    # Create service definition draft
    arcpy.mapping.CreateMapSDDraft(mxd, sddraft, service, 'ARCGIS_SERVER', con, copy_data_to_server, serviceFolder)
    # Analyze the service definition draft
    analysis = arcpy.mapping.AnalyzeForSD(sddraft)

    # Print errors, warnings, and messages returned from the analysis
    print "The following information was returned during analysis of the MXD:"
    for key in ('messages', 'warnings', 'errors'):
        print '----' + key.upper() + '---'
        vars = analysis[key]
        for ((message, code), layerlist) in vars.iteritems():
            print '    ', message, ' (CODE %i)' % code
            print '       applies to:',
            for layer in layerlist:
                print layer.name,
            print

    # Stage and upload the service if the sddraft analysis did not contain errors
    if analysis['errors'] == {}:
        # Execute StageService. This creates the service definition.
        arcpy.StageService_server(sddraft, sd)
        # Execute UploadServiceDefinition. This uploads the service definition and publishes the service.
        # 函数说明：https://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/server-toolbox/upload-service-definition.htm
        # arcpy.UploadServiceDefinition_server(sd, con)
        # print "==============================================="
        # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis")
        # print "登录门户成功"
        # print "==============================================="
        # Set local variables
        inSdFile = "myMapService.sd"
        inServer = "myServerConnection.ags"
        inServiceName = ""
        inCluster = ""
        inFolderType = ""
        inFolder = ""
        inStartup = ""
        inOverride = "OVERRIDE_DEFINITION"
        inMyContents = "SHARE_ONLINE"
        inPublic = "PRIVATE"
        inOrganization = "NO_SHARE_ORGANIZATION"
        inGroups = "My Group"

        # Execute UploadServiceDefinition
        # arcpy.UploadServiceDefinition_server(sd, con, inServiceName,
        #                                      inCluster, inFolderType, inFolder,
        #                                      inStartup, inOverride, inMyContents,
        #                                      inPublic, inOrganization, inGroups)
        # arcpy.SharePackage_management()
        # arcpy.UploadServiceDefinition_server(sd, con,"AAA",None,None,None,None,None,"SHARE_ONLINE","PUBLIC","SHARE_ORGANIZATION",None)
        arcpy.UploadServiceDefinition_server(sd, con,"A02")
        # arcpy.UploadServiceDefinition_server(sd,con,"AAA","","","","","OVERRIDE_DEFINITION","SHARE_ONLINE","PUBLIC","SHARE_ORGANIZATION","")
        # arcpy.UploadServiceDefinition_server(sd, con, "AAA",
        #                                      "", "", "", "", "OVERRIDE_DEFINITION", "SHARE_ONLINE",
        #                                      "PUBLIC", "SHARE_ORGANIZATION", "")
        print "Service successfully published"
    else:
        print "Service could not be published because errors were found during analysis."

    print arcpy.GetMessages()

# demoMXDPath：包含mxd文档名称
# folder：包含新建的mxd文档以及tiff文件的文件夹路径
def createMxdDocument(demoMXDPath, folder,serviceDir):
    if os.path.exists(demoMXDPath) == False:
        print "mxd document it's not exist!"
    else:
        try:
            print "opening mxd document……"
            # print "==============================================="
            # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", serviceDir)
            # print "==============================================="
            mxd = arcpy.mapping.MapDocument(demoMXDPath)
            print "repair layer source"
            if os.path.isdir(folder) == False:
                print "invalid document path!"
                return
            print "reading layer document one by one......"
            files = os.listdir(folder)
            i = 0
            layerName = ""
            for f in files:
                if f.endswith(".tif"):
                    if layerName == "":
                        name1 = f.replace("nasa-worldview-", "")
                        layerName = name1[0:9]
                    if i > 3:
                            continue
                    if f.index(layerName) >= 0:
                        print folder
                        # 构建金字塔处理tif
                        # tifflist = arcpy.ListRasters("", "TIF")
                        # for tiff in tifflist:
                        #     arcpy.BuildPyramids_management(tiff)
                        arcpy.env.pyramid = "PYRAMIDS -1 BILINEAR DEFAULT"
                        arcpy.BuildPyramids_management(os.path.join(folder,f))

                        print layerName + ";" + f
                        # 获取图层
                        # df = arcpy.mapping.ListDataFrames(mxd, "Layers")[0]
                        df = arcpy.mapping.ListDataFrames(mxd, "Layers")
                        print arcpy.mapping.ListLayers(mxd, "", df)[i].name
                        # 对图层进行修改
                        lyr = arcpy.mapping.ListLayers(mxd, "", df)[i]
                        # 设置数据源
                        lyr.replaceDataSource(folder, "RASTER_WORKSPACE", f)
                        lyr.name = f.replace(".tif", "")
                        i = i + 1
                else:
                    continue

            mxdName = time.strftime("%Y_%m_%d", time.localtime()) + "_1_"+ files[0].split(".")[0] + ".mxd"  # 2015_11_24样式文件名
            newMXD = folder + "\\" + mxdName
            mxd.saveACopy(newMXD)
            del mxd

        except Exception, e:
            print "open mxd error: ", e
            return

if __name__ == '__main__':
    # print "==============================================="
    # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis")
    # print "登录门户成功"
    # print "==============================================="
    # print "==============================================="
    # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis/home")
    # # arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com:6443/arcgis/manager")
    # print arcpy.GetMessages()
    # print arcpy.SignInToPortal_server("gisgis", "1qazxsw2", "https://arcgis.ygwl.com/arcgis/home")
    # print "登录门户成功"
    # print "==============================================="
    tiffFolder=time.strftime("%Y_%m_%d", time.localtime())
    print tiffFolder
    folderPath=r"E:\pycharm_project\tfservingconvert\gis\tiff"
    serviceDir = "https://arcgis.ygwl.com/arcgis"
    createMxdDocument(r"E:\pycharm_project\tfservingconvert\gis\gis\dingan.mxd",folderPath,serviceDir)
    # PublishAll(folderPath, "https://arcgis.ygwl.com/arcgis", "ygwl")
    PublishAll(folderPath,serviceDir ,"")
    print "end."